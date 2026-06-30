"""build_graph()/apply_graph() split -- new coverage.

forward() was refactored to call these two methods internally so the
graph could be cached across layers (see test_cached_graph_transformer.py
and CachedGraphSparseTransformer). This file verifies the split itself
is correct: calling build_graph then apply_graph manually must produce
bit-identical output to calling forward() directly, for both
self-attention and cross-attention.
"""
import torch

from sparse_attention import SSAConfig, SparseAttention


def test_split_path_matches_forward_self_attention(base_cfg):
    attn = SparseAttention(base_cfg)
    attn.eval()
    x = torch.randn(2, 48, base_cfg.d_model)

    with torch.no_grad():
        out_direct, _ = attn(x)

        q = attn._split_q(attn.Wq(x))
        k = attn._split_kv(attn.Wk(x))
        v = attn._split_kv(attn.Wv(x))
        neighbors, valid = attn.build_graph(
            q, k, N=48, M=48, B=2, device=x.device, causal=False, is_cross=False
        )
        out_split = attn.Wo(attn._merge(attn.apply_graph(q, k, v, neighbors, valid)))

    assert torch.allclose(out_direct, out_split, atol=1e-6)


def test_split_path_matches_forward_cross_attention(base_cfg):
    attn = SparseAttention(base_cfg)
    attn.eval()
    qx = torch.randn(1, 40, base_cfg.d_model)
    kv = torch.randn(1, 80, base_cfg.d_model)

    with torch.no_grad():
        out_direct, _ = attn(qx, key_value=kv)

        q = attn._split_q(attn.Wq(qx))
        k = attn._split_kv(attn.Wk(kv))
        v = attn._split_kv(attn.Wv(kv))
        neighbors, valid = attn.build_graph(
            q, k, N=40, M=80, B=1, device=qx.device, causal=False, is_cross=True
        )
        out_split = attn.Wo(attn._merge(attn.apply_graph(q, k, v, neighbors, valid)))

    assert torch.allclose(out_direct, out_split, atol=1e-6)


def test_split_path_matches_forward_causal(causal_cfg):
    attn = SparseAttention(causal_cfg)
    attn.eval()
    x = torch.randn(1, 32, causal_cfg.d_model)

    with torch.no_grad():
        out_direct, _ = attn(x)

        q = attn._split_q(attn.Wq(x))
        k = attn._split_kv(attn.Wk(x))
        v = attn._split_kv(attn.Wv(x))
        neighbors, valid = attn.build_graph(
            q, k, N=32, M=32, B=1, device=x.device, causal=True, is_cross=False
        )
        out_split = attn.Wo(attn._merge(attn.apply_graph(q, k, v, neighbors, valid)))

    assert torch.allclose(out_direct, out_split, atol=1e-6)


def test_apply_graph_reused_against_different_qkv_does_not_crash():
    """Sanity check for the use case CachedGraphSparseTransformer relies
    on: a graph built from one set of q/k can be applied to a DIFFERENT
    layer's q/k/v of the same shape without shape errors. This does not
    assert anything about output QUALITY (see test_cached_graph_transformer
    and the README's honesty caveat about cross-layer reuse) -- only that
    the mechanism itself doesn't crash or shape-mismatch."""
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    attn_a = SparseAttention(cfg)
    attn_b = SparseAttention(cfg)
    x_a = torch.randn(1, 20, 32)
    x_b = torch.randn(1, 20, 32)

    q_a = attn_a._split_q(attn_a.Wq(x_a))
    k_a = attn_a._split_kv(attn_a.Wk(x_a))
    neighbors, valid = attn_a.build_graph(
        q_a, k_a, N=20, M=20, B=1, device=x_a.device, causal=False, is_cross=False
    )

    q_b = attn_b._split_q(attn_b.Wq(x_b))
    k_b = attn_b._split_kv(attn_b.Wk(x_b))
    v_b = attn_b._split_kv(attn_b.Wv(x_b))
    out = attn_b.apply_graph(q_b, k_b, v_b, neighbors, valid)

    assert out.shape == q_b.shape
    assert not out.isnan().any()
