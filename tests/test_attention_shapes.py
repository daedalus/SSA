"""Basic forward-pass shape and contract tests.

Ported from the original __main__ checks 1 and 2.
"""
import torch

from sparse_attention import SparseAttention, SparseTransformer


def test_self_attention_shape(base_cfg):
    model = SparseTransformer(base_cfg, num_layers=2)
    x = torch.randn(2, 64, base_cfg.d_model)
    out, stats = model(x, return_stats=True)

    assert out.shape == x.shape
    s = stats[0]
    assert s["neighbors_shape"] == (2, base_cfg.num_heads, 64, base_cfg.num_neighbors)
    assert s["compression"] > 1.0


def test_self_attention_return_stats_default_empty(base_cfg):
    """return_stats=False (the default) must return an empty dict, not
    a populated one -- this is a deliberate contract, not an oversight."""
    model = SparseTransformer(base_cfg, num_layers=2)
    x = torch.randn(2, 64, base_cfg.d_model)
    out, stats = model(x)

    assert out.shape == x.shape
    assert all(s == {} for s in stats)


def test_cross_attention_shape(base_cfg):
    attn = SparseAttention(base_cfg)
    qx = torch.randn(1, 48, base_cfg.d_model)
    kv = torch.randn(1, 96, base_cfg.d_model)
    out, stats = attn(qx, key_value=kv, return_stats=True)

    assert out.shape == qx.shape
    assert stats["neighbors_shape"][:3] == (1, base_cfg.num_heads, 48)


def test_cross_attention_neighbor_indices_in_range(base_cfg):
    """All selected key indices for cross-attention must be < M (the key
    sequence length), never N (the query sequence length) -- these can
    differ and a stale clamp bug previously let query-length-sized
    indices leak through."""
    attn = SparseAttention(base_cfg)
    attn.eval()
    qx = torch.randn(1, 48, base_cfg.d_model)
    kv = torch.randn(1, 96, base_cfg.d_model)

    with torch.no_grad():
        q = attn._split_q(attn.Wq(qx))
        k = attn._split_kv(attn.Wk(kv))
        neighbors, valid = attn.build_graph(
            q, k, N=48, M=96, B=1, device="cpu", causal=False, is_cross=True
        )

    assert (neighbors[valid] < 96).all()
    assert (neighbors[valid] >= 0).all()
