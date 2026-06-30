"""Explicit global_token_indices and the attention_mask rejection guard.

New coverage -- both features were added in a later commit alongside
GQA/MQA and had no real test coverage.
"""
import pytest
import torch

from sparse_attention import SSAConfig, SparseAttention, GlobalGraphBuilder


def test_global_token_indices_overrides_num_global_tokens():
    cfg = SSAConfig(d_model=64, num_heads=2, num_neighbors=16,
                     window_size=4, num_global_tokens=2,
                     global_token_indices=[0, 15, 31], max_num_hashes=3)
    attn = SparseAttention(cfg)
    x = torch.randn(2, 32, 64)

    out, _ = attn(x)
    assert out.shape == x.shape


def test_global_graph_builder_explicit_indices_returned_verbatim(device):
    builder = GlobalGraphBuilder(num_global=2, indices=[0, 5, 17])
    idx = builder(M=32, device=device)

    assert idx.tolist() == [0, 5, 17]


def test_global_graph_builder_clamps_out_of_range_indices(device):
    """If a config was built for a longer sequence and applied to a
    shorter one, explicit global indices must clamp to the valid range
    rather than producing an out-of-bounds index downstream."""
    builder = GlobalGraphBuilder(num_global=2, indices=[0, 100, 200])
    idx = builder(M=32, device=device)

    assert (idx < 32).all()
    assert (idx >= 0).all()


def test_global_graph_builder_default_mode_unchanged(device):
    """When indices=None, behavior must match the original
    first-G-positions default exactly -- this is a regression guard for
    the refactor that added the indices parameter."""
    builder = GlobalGraphBuilder(num_global=3, indices=None)
    idx = builder(M=32, device=device)

    assert idx.tolist() == [0, 1, 2]


def test_attention_mask_raises_value_error(base_cfg):
    attn = SparseAttention(base_cfg)
    x = torch.randn(2, 32, base_cfg.d_model)
    mask = torch.ones(2, 32)

    with pytest.raises(ValueError, match="attention_mask"):
        attn(x, attention_mask=mask)


def test_attention_mask_none_does_not_raise(base_cfg):
    attn = SparseAttention(base_cfg)
    x = torch.randn(2, 32, base_cfg.d_model)

    out, _ = attn(x, attention_mask=None)
    assert out.shape == x.shape
