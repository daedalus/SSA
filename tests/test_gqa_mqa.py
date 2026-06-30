"""GQA/MQA support -- new coverage, not present in the original
__main__ test block. These features were added in a later commit and
had only ad-hoc smoke testing, not real test coverage.
"""
import pytest
import torch

from sparse_attention import SSAConfig, SparseAttention


def test_mqa_output_shape(mqa_cfg):
    attn = SparseAttention(mqa_cfg)
    x = torch.randn(2, 32, mqa_cfg.d_model)
    out, _ = attn(x)
    assert out.shape == x.shape


def test_gqa_output_shape(gqa_cfg):
    attn = SparseAttention(gqa_cfg)
    x = torch.randn(2, 32, gqa_cfg.d_model)
    out, _ = attn(x)
    assert out.shape == x.shape


def test_mqa_has_fewer_kv_params_than_mha():
    mha_cfg = SSAConfig(d_model=64, num_heads=4, num_neighbors=16,
                         window_size=4, num_global_tokens=2, max_num_hashes=3)
    mqa_cfg = SSAConfig(d_model=64, num_heads=4, num_kv_heads=1, num_neighbors=16,
                         window_size=4, num_global_tokens=2, max_num_hashes=3)

    mha_attn = SparseAttention(mha_cfg)
    mqa_attn = SparseAttention(mqa_cfg)

    mha_kv_params = sum(p.numel() for p in mha_attn.Wk.parameters()) + \
                    sum(p.numel() for p in mha_attn.Wv.parameters())
    mqa_kv_params = sum(p.numel() for p in mqa_attn.Wk.parameters()) + \
                    sum(p.numel() for p in mqa_attn.Wv.parameters())

    # 4 query heads sharing 1 KV head -> exactly 4x fewer KV params
    assert mqa_kv_params * 4 == mha_kv_params


def test_gqa_kv_params_scale_with_kv_head_count():
    """num_kv_heads=2 with num_heads=4 should land exactly between full
    MHA and MQA in KV parameter count."""
    cfg2 = SSAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_neighbors=16,
                      window_size=4, num_global_tokens=2, max_num_hashes=3)
    cfg1 = SSAConfig(d_model=64, num_heads=4, num_kv_heads=1, num_neighbors=16,
                      window_size=4, num_global_tokens=2, max_num_hashes=3)
    cfg4 = SSAConfig(d_model=64, num_heads=4, num_kv_heads=4, num_neighbors=16,
                      window_size=4, num_global_tokens=2, max_num_hashes=3)

    def kv_params(cfg):
        attn = SparseAttention(cfg)
        return sum(p.numel() for p in attn.Wk.parameters()) + \
               sum(p.numel() for p in attn.Wv.parameters())

    p1, p2, p4 = kv_params(cfg1), kv_params(cfg2), kv_params(cfg4)
    assert p1 < p2 < p4
    assert p2 * 2 == p4
    assert p1 * 4 == p4


def test_num_heads_must_be_divisible_by_num_kv_heads():
    with pytest.raises(AssertionError):
        SparseAttention(SSAConfig(d_model=64, num_heads=5, num_kv_heads=2))


def test_gqa_gradient_flow():
    cfg = SSAConfig(d_model=64, num_heads=4, num_kv_heads=2, num_neighbors=16,
                     window_size=4, num_global_tokens=2, max_num_hashes=3)
    attn = SparseAttention(cfg)
    x = torch.randn(2, 24, 64, requires_grad=True)

    out, _ = attn(x)
    out.sum().backward()

    assert not x.grad.isnan().any()
    assert attn.Wk.weight.grad is not None
    assert attn.Wk.weight.grad.norm().item() > 0.0
