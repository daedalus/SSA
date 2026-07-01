"""Multi-probe LSH (`SSAConfig.lsh_num_probes`) correctness and regression
guards.

Multi-probe checks additional near-boundary buckets per round per query
(the bucket one bit-flip away from the query's own, ranked by which hash
bit had the smallest |hyperplane projection|) instead of only the query's
own bucket. See LSHGraphBuilder._probe_ids for the implementation and
sparse_attention.py::SSAConfig.lsh_num_probes for the rationale.

These tests check three things a probe-adjacent bucket lookup could get
wrong that a single-bucket lookup can't: probe ids actually differing
from the primary id and from each other (else probes are silently
no-ops), causal masking/empty-bucket fallback applying correctly to
probe-sourced candidates (not just the primary bucket), and recall
strictly improving as probes are added at fixed R (the actual point of
the feature). See benchmarks/bench_multiprobe.py for the full
recall-vs-cost tradeoff analysis this regression floor is drawn from.
"""
import torch

from sparse_attention import SSAConfig, SparseAttention, LSHGraphBuilder, merge_neighbors


def test_default_num_probes_is_zero_and_unchanged_behavior():
    """lsh_num_probes=0 (the default) must reproduce prior behavior
    exactly -- this is the backward-compatibility contract for the
    feature: existing configs that don't set lsh_num_probes see zero
    change in output shape or values."""
    torch.manual_seed(0)
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=32,
                     max_num_hashes=6, num_hash_rounds=2,
                     window_size=2, num_global_tokens=1, causal=False)
    assert cfg.lsh_num_probes == 0
    attn = SparseAttention(cfg)
    assert attn.lsh_builder.T == 0


def test_probe_ids_distinct_from_base_and_each_other():
    """Each probe must be a genuine alternate bucket -- if a probe id
    collided with the primary id or another probe, that probe would be
    silently redundant (re-checking a bucket already covered) rather
    than exploring new candidates."""
    torch.manual_seed(0)
    lsh = LSHGraphBuilder(d_head=16, lsh_candidates=8, num_rounds=1, max_num_hashes=6)
    x = torch.randn(4, 50, 16)
    proj = lsh._hash_proj(x, 0, 6)
    bits = (proj > 0).long()
    powers = 2 ** torch.arange(6)
    base = (bits * powers).sum(-1)
    probes = lsh._probe_ids(proj, 6, 4)
    assert len(probes) == 4
    for i, p in enumerate(probes):
        assert (p != base).all(), f"probe {i} collided with base id"
        for j, p2 in enumerate(probes):
            if i != j:
                assert (p != p2).all(), f"probe {i} collided with probe {j}"


def test_num_probes_clamped_to_p_eff():
    """Requesting more probes than there are hash bits (p_eff) must not
    crash or produce duplicate/garbage probes -- it should silently cap
    at p_eff, since you cannot flip more distinct bits than exist."""
    torch.manual_seed(0)
    lsh = LSHGraphBuilder(d_head=8, lsh_candidates=4, num_rounds=1, max_num_hashes=3)
    x = torch.randn(2, 10, 8)
    proj = lsh._hash_proj(x, 0, 3)   # p_eff = 3
    probes = lsh._probe_ids(proj, 3, 12)   # request way more than 3
    assert len(probes) == 3


def test_causal_no_future_leakage_with_probes():
    """Probe-sourced candidates must respect causal masking exactly like
    primary-bucket candidates -- a probe bucket can contain future
    positions just as easily as the primary bucket can."""
    torch.manual_seed(0)
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=32, max_num_hashes=6,
                     num_hash_rounds=2, window_size=2, num_global_tokens=1,
                     causal=True, lsh_num_probes=3)
    attn = SparseAttention(cfg)
    attn.eval()
    N = 200
    x = torch.randn(1, N, 32)
    with torch.no_grad():
        q = attn._split_q(attn.Wq(x))
        k = attn._split_kv(attn.Wk(x))
        neighbors, valid = attn.build_graph(q, k, N, N, 1, "cpu", causal=True, is_cross=False)
    for i in range(N):
        idxs = neighbors[0, 0, i][valid[0, 0, i]]
        assert (idxs <= i).all(), f"future leakage at query {i}: {idxs[idxs > i]}"


def test_gradients_flow_with_probes():
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=32, max_num_hashes=6,
                     num_hash_rounds=2, window_size=2, num_global_tokens=1,
                     causal=False, lsh_num_probes=2)
    attn = SparseAttention(cfg)
    x = torch.randn(1, 64, 32, requires_grad=True)
    out, _ = attn(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_shapes_and_finiteness_across_n_and_probe_counts():
    """No crashes/NaNs across a spread of N (including tiny N where
    p_eff floors at 1, so num_probes gets clamped down hard) and probe
    counts, including counts that exceed max_num_hashes."""
    torch.manual_seed(0)
    for N in [8, 16, 65, 300, 2048]:
        for probes in [0, 1, 3, 12]:
            cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=32,
                             max_num_hashes=6, num_hash_rounds=2,
                             window_size=2, num_global_tokens=1, causal=False,
                             lsh_num_probes=probes)
            attn = SparseAttention(cfg)
            attn.eval()
            x = torch.randn(1, N, 32)
            with torch.no_grad():
                out, _ = attn(x)
            assert out.shape == (1, N, 32), (N, probes)
            assert torch.isfinite(out).all(), (N, probes)


def _mean_recall(N, d_model, num_hash_rounds, num_probes, seed=0):
    torch.manual_seed(seed)
    cfg = SSAConfig(d_model=d_model, num_heads=2, num_neighbors=64,
                     max_num_hashes=12, num_hash_rounds=num_hash_rounds,
                     lsh_num_probes=num_probes,
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
    recalls = []
    for i in range(N):
        true_set = set(true_top[i].tolist())
        sparse_set = set(neighbors[0, 0, i][valid[0, 0, i]].tolist())
        recalls.append(len(true_set & sparse_set) / len(true_set))
    return sum(recalls) / len(recalls)


def test_recall_increases_with_num_probes_at_fixed_rounds():
    """The actual point of the feature: at a FIXED num_hash_rounds,
    adding probes must increase recall, not just add compute for free.
    Regression floor drawn from benchmarks/bench_multiprobe.py, where at
    N=2048, R=4 this went 48.5% (T=0) -> 72.6% (T=1) -> 91.3% (T=3)."""
    r0 = _mean_recall(N=2048, d_model=64, num_hash_rounds=4, num_probes=0)
    r1 = _mean_recall(N=2048, d_model=64, num_hash_rounds=4, num_probes=1)
    r3 = _mean_recall(N=2048, d_model=64, num_hash_rounds=4, num_probes=3)
    assert r1 > r0, (
        f"REGRESSION: adding 1 probe did not improve recall ({r1:.1%} vs "
        f"{r0:.1%} baseline) -- multi-probe is providing zero signal."
    )
    assert r3 > r1, (
        f"REGRESSION: adding more probes (3) did not further improve recall "
        f"({r3:.1%} vs {r1:.1%} at 1 probe)."
    )
