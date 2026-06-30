"""CachedGraphSparseTransformer and the neighbor_overlap/
measure_cross_layer_overlap diagnostic utilities -- new coverage.

These were added alongside the LSA-inspired cross-layer graph caching
feature and previously only had ad-hoc smoke testing in a throwaway
script, not real pytest coverage.
"""
import torch

from sparse_attention import (
    SSAConfig,
    SparseTransformer,
    CachedGraphSparseTransformer,
    neighbor_overlap,
    measure_cross_layer_overlap,
)


def test_cached_graph_transformer_output_shape(base_cfg):
    model = CachedGraphSparseTransformer(base_cfg, num_layers=6, reuse_every=3)
    x = torch.randn(2, 48, base_cfg.d_model)

    out, stats = model(x, return_stats=True)
    assert out.shape == x.shape
    assert len(stats) == 6


def test_cached_graph_transformer_recompute_pattern(base_cfg):
    """With reuse_every=3 across 6 layers, the graph should be rebuilt
    exactly at the start of each group: layers 0 and 3, and reused for
    layers 1, 2, 4, 5."""
    model = CachedGraphSparseTransformer(base_cfg, num_layers=6, reuse_every=3)
    x = torch.randn(2, 48, base_cfg.d_model)

    _, stats = model(x, return_stats=True)
    recompute_flags = [s["graph_recomputed"] for s in stats]

    assert recompute_flags == [True, False, False, True, False, False]


def test_cached_graph_transformer_reuse_every_one_always_recomputes(base_cfg):
    """reuse_every=1 is the degenerate case -- every layer rebuilds its
    own graph, equivalent in spirit to the uncached SparseTransformer."""
    model = CachedGraphSparseTransformer(base_cfg, num_layers=4, reuse_every=1)
    x = torch.randn(1, 32, base_cfg.d_model)

    _, stats = model(x, return_stats=True)
    recompute_flags = [s["graph_recomputed"] for s in stats]

    assert all(recompute_flags)


def test_cached_graph_transformer_gradient_flow(base_cfg):
    model = CachedGraphSparseTransformer(base_cfg, num_layers=4, reuse_every=2)
    x = torch.randn(1, 32, base_cfg.d_model, requires_grad=True)

    out, _ = model(x)
    out.sum().backward()

    assert not x.grad.isnan().any()
    assert x.grad.norm().item() > 0.0


def test_neighbor_overlap_identical_sets_is_one():
    B, H, N, K = 1, 2, 8, 4
    idx = torch.randint(0, 16, (B, H, N, K))
    valid = torch.ones(B, H, N, K, dtype=torch.bool)

    overlap = neighbor_overlap(idx, idx, valid, valid)
    assert torch.allclose(overlap, torch.ones_like(overlap))


def test_neighbor_overlap_disjoint_sets_is_zero():
    B, H, N, K = 1, 2, 8, 4
    idx_a = torch.arange(K).view(1, 1, 1, K).expand(B, H, N, K).clone()
    idx_b = idx_a + 1000  # guaranteed disjoint from idx_a
    valid = torch.ones(B, H, N, K, dtype=torch.bool)

    overlap = neighbor_overlap(idx_a, idx_b, valid, valid)
    assert torch.allclose(overlap, torch.zeros_like(overlap))


def test_neighbor_overlap_partial_overlap_between_zero_and_one():
    B, H, N, K = 1, 1, 1, 4
    idx_a = torch.tensor([[[[0, 1, 2, 3]]]])
    idx_b = torch.tensor([[[[2, 3, 4, 5]]]])  # 2 of 4 shared -> jaccard 2/6
    valid = torch.ones(B, H, N, K, dtype=torch.bool)

    overlap = neighbor_overlap(idx_a, idx_b, valid, valid)
    assert abs(overlap.item() - (2.0 / 6.0)) < 1e-6


def test_neighbor_overlap_respects_invalid_mask():
    """Padding/invalid slots must never be counted as a match -- a
    sentinel collision (e.g. both sides padding with the same dummy
    value) would silently inflate the overlap score if not masked out."""
    B, H, N, K = 1, 1, 1, 4
    idx_a = torch.tensor([[[[0, 1, 0, 0]]]])  # last 2 slots are padding
    idx_b = torch.tensor([[[[0, 1, 0, 0]]]])  # same raw values
    valid_a = torch.tensor([[[[True, True, False, False]]]])
    valid_b = torch.tensor([[[[True, True, False, False]]]])

    overlap = neighbor_overlap(idx_a, idx_b, valid_a, valid_b)
    # Only slots 0,1 are valid on both sides and they match exactly -> 1.0
    assert torch.allclose(overlap, torch.ones_like(overlap))


def test_measure_cross_layer_overlap_runs_and_returns_expected_shape(base_cfg):
    model = SparseTransformer(base_cfg, num_layers=4)
    x = torch.randn(2, 32, base_cfg.d_model)

    report = measure_cross_layer_overlap(model, x)

    assert len(report["mean_overlap_per_adjacent_pair"]) == 3  # 4 layers -> 3 pairs
    assert report["num_layers"] == 4
    for v in report["mean_overlap_per_adjacent_pair"]:
        assert 0.0 <= v <= 1.0


def test_measure_cross_layer_overlap_on_untrained_model_is_not_near_one(base_cfg):
    """Documents and guards the honesty claim made in the README: on an
    untrained model, cross-layer neighbor overlap should NOT be high,
    because LSHGraphBuilder's hash planes are independently initialized
    per layer with no training signal pushing them toward similarity.
    If this regresses to consistently measuring near 1.0, something
    about the graph construction has accidentally become
    layer-independent (e.g. a shared random seed bug), which would make
    the whole point of this diagnostic moot."""
    model = SparseTransformer(base_cfg, num_layers=4)
    x = torch.randn(2, 32, base_cfg.d_model)

    report = measure_cross_layer_overlap(model, x)
    mean_overlap = sum(report["mean_overlap_per_adjacent_pair"]) / len(
        report["mean_overlap_per_adjacent_pair"]
    )

    assert mean_overlap < 0.9, (
        f"untrained-model cross-layer overlap measured {mean_overlap:.3f}, "
        f"unexpectedly close to 1.0 -- this would undermine the documented "
        f"caveat that cross-layer graph caching is unsafe to assume by default"
    )
