"""Recall@K vs context length benchmark.

Extends tests/test_quality_and_scaling.py's single-N (N=1024) recall
check into a sweep across context lengths, to answer: does the LSH
routing's "recall capacity" hold up as N grows, or does it degrade?

For each N, this builds the actual neighbor graph SparseAttention would
use (window + global + LSH, deduplicated/merged exactly like forward()
does), and measures:

  - recall@true_k  : overlap between the sparse candidate set and exact
                      dense top-K, averaged over all query positions
  - random floor    : recall of a uniformly random candidate set of the
                      same size (the "would a dumb baseline also get
                      this by luck" control)
  - ratio           : recall / random floor (this is the number that
                      should stay roughly flat or grow with N if the
                      adaptive hash-plane-count logic in LSHGraphBuilder
                      is doing its job -- since num_neighbors is fixed
                      while N grows, the random floor shrinks like
                      K/N, so a *falling* ratio despite that shrinking
                      floor would mean the LSH signal itself is
                      degrading, not just being diluted)
  - wall time       : cost of building the graph at this N (LSH +
                      window + global + dedup/merge), separate from
                      attention compute
"""
import time

import torch

from sparse_attention import SSAConfig, SparseAttention, merge_neighbors

CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
# 16384 excluded by default: on a memory-constrained host (this sandbox
# has ~3.9GB RAM) the exact-dense-scores reference computation plus LSH
# bucket tensors OOM. Pass --big to attempt it anyway on a larger machine.
D_MODEL = 64
NUM_NEIGHBORS = 64
TRUE_K = 32
SEED = 0


def build_graph_and_measure(N: int):
    torch.manual_seed(SEED)
    cfg = SSAConfig(
        d_model=D_MODEL, num_heads=2, num_neighbors=NUM_NEIGHBORS,
        max_num_hashes=12, num_hash_rounds=4,
        window_size=8, num_global_tokens=4, causal=False,
    )
    attn = SparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, D_MODEL)

    t0 = time.perf_counter()
    with torch.no_grad():
        q = attn._split_q(attn.Wq(x))
        k = attn._split_kv(attn.Wk(x))
        glob_idx = attn.glob_builder(N, "cpu")
        win_idx = attn.win_builder(N, N, "cpu")
        lsh_idx = attn.lsh_builder(q, k, attn.lsh_k, False, glob_idx)
        self_idx = torch.arange(N).view(1, 1, N).expand(1, 2, N).clone()
        neighbors, valid = merge_neighbors(
            win_idx, glob_idx, lsh_idx, self_idx, cfg.num_neighbors,
            N, 1, 2, "cpu",
        )
    build_s = time.perf_counter() - t0

    q0, k0 = q[0, 0], k[0, 0]
    dense_scores = q0 @ k0.T
    true_k = min(TRUE_K, N - 1)
    _, true_top = dense_scores.topk(true_k, dim=-1)

    torch.manual_seed(SEED + 1)
    recalls, random_recalls = [], []
    for i in range(N):
        true_set = set(true_top[i].tolist())
        sparse_set = set(neighbors[0, 0, i][valid[0, 0, i]].tolist())
        recalls.append(len(true_set & sparse_set) / len(true_set))
        rand_set = set(torch.randperm(N)[:cfg.num_neighbors].tolist())
        random_recalls.append(len(true_set & rand_set) / len(true_set))

    mean_recall = sum(recalls) / len(recalls)
    mean_random = sum(random_recalls) / len(random_recalls)
    ratio = mean_recall / mean_random if mean_random > 0 else float("inf")

    return {
        "N": N,
        "recall": mean_recall,
        "random_floor": mean_random,
        "ratio": ratio,
        "build_s": build_s,
    }


def main():
    import sys
    lengths = list(CONTEXT_LENGTHS)
    if "--big" in sys.argv:
        lengths.append(16384)

    print(f"{'N':>7} | {'recall@K':>9} | {'random floor':>12} | {'ratio':>7} | {'graph build (s)':>16}")
    print("-" * 65)
    results = []
    for N in lengths:
        r = build_graph_and_measure(N)
        results.append(r)
        print(f"{r['N']:>7} | {r['recall']:>8.1%} | {r['random_floor']:>11.2%} | "
              f"{r['ratio']:>6.2f}x | {r['build_s']:>16.3f}")

    ratios = [r["ratio"] for r in results]
    print()
    if ratios == sorted(ratios):
        print("Ratio vs random floor is non-decreasing across context lengths "
              "-- recall capacity holds up as N grows.")
    elif ratios[-1] < ratios[0] * 0.5:
        print("WARNING: ratio vs random floor drops sharply at long context "
              "-- LSH signal is degrading faster than the shrinking random "
              "floor alone would explain.")
    else:
        print("Ratio vs random floor is roughly stable, with some "
              "fluctuation, across context lengths.")

    return results


if __name__ == "__main__":
    main()
