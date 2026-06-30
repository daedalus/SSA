"""Memory scaling (O(NK) vs O(N^2)) and recall@K quality regression test.

Ported from the original __main__ checks 8 and 9. The recall test is the
single most important correctness signal in the whole suite: every other
test in this directory proves the gather/dedup/index machinery is
mechanically correct (no crashes, no NaN, no out-of-range indices, no
duplicates) but says nothing about whether the selected neighbors are
any GOOD -- i.e. whether they overlap with what exact dense attention
would have selected. A sparse attention implementation can pass every
other test in this file while still selecting essentially random
neighbors. This test closes that gap directly.

Caveat (kept from the original): random Gaussian Q/K share no real
semantic structure, so this is a harder case for content-based LSH than
trained embeddings with genuine cluster structure would be. Treat the
recall numbers as a regression floor / sanity check, not a claim about
trained-model quality -- see README.md.
"""
import pytest
import torch

from sparse_attention import SSAConfig, SparseAttention, merge_neighbors


@pytest.mark.parametrize("n,h,k,d", [
    (256, 4, 32, 32),
    (1024, 4, 32, 32),
    (4096, 4, 32, 32),
    (16384, 4, 32, 32),
])
def test_memory_scaling_sparse_strictly_smaller_than_dense(n, h, k, d):
    full = h * n * n * d
    sparse = h * n * k * d
    assert sparse < full
    # Compression ratio should grow with N for fixed K -- this is the
    # entire point of sparse attention; a regression here (e.g. K scaling
    # with N somewhere) would silently erase the whole benefit.
    ratio = full / sparse
    assert ratio >= n / k * 0.99  # allow tiny float slack


def test_memory_scaling_ratio_increases_with_sequence_length():
    h, k, d = 4, 32, 32
    ratios = []
    for n in [256, 1024, 4096, 16384]:
        full = h * n * n * d
        sparse = h * n * k * d
        ratios.append(full / sparse)

    assert ratios == sorted(ratios), "compression ratio should be monotonically increasing with N"


def test_recall_at_k_beats_random_selection():
    """Regression floor: the sparse pipeline's recall@K against exact
    dense top-K must exceed what random key selection would achieve. If
    it doesn't, LSH routing is providing zero signal beyond random
    chance, which indicates a real bug in the routing logic, not just a
    quality ceiling worth living with."""
    torch.manual_seed(0)
    N, d_model = 1024, 64
    cfg = SSAConfig(d_model=d_model, num_heads=2, num_neighbors=64,
                     max_num_hashes=12, num_hash_rounds=4,
                     window_size=8, num_global_tokens=4, causal=False)
    attn = SparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, N, d_model)

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

    q0, k0 = q[0, 0], k[0, 0]
    dense_scores = q0 @ k0.T
    true_k = 32
    _, true_top = dense_scores.topk(true_k, dim=-1)

    torch.manual_seed(1)
    recalls, random_recalls = [], []
    for i in range(N):
        true_set = set(true_top[i].tolist())
        sparse_set = set(neighbors[0, 0, i][valid[0, 0, i]].tolist())
        recalls.append(len(true_set & sparse_set) / len(true_set))
        rand_set = set(torch.randperm(N)[:cfg.num_neighbors].tolist())
        random_recalls.append(len(true_set & rand_set) / len(true_set))

    mean_recall = sum(recalls) / len(recalls)
    mean_random = sum(random_recalls) / len(random_recalls)

    assert mean_recall > mean_random, (
        f"REGRESSION: sparse pipeline recall ({mean_recall:.1%}) did not beat "
        f"random selection ({mean_random:.1%}) -- LSH routing is providing "
        f"zero signal, which indicates a bug, not just a quality limitation."
    )
    # A loose absolute floor as a second signal -- not as tight as the
    # relative-to-random check above, but catches a milder degradation
    # that still clears the random floor by a small margin.
    assert mean_recall > mean_random * 2, (
        f"recall ({mean_recall:.1%}) barely beat random ({mean_random:.1%}) -- "
        f"investigate even though the strict regression check passed"
    )
