"""Gradient flow correctness -- ensures the sparse gather/scatter
machinery doesn't break autograd or produce NaN gradients.

Ported from the original __main__ check 7.
"""
import torch

from sparse_attention import SSAConfig, SparseTransformer, SparseAttention


def test_gradient_flow_no_nan():
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    model = SparseTransformer(cfg, num_layers=2)
    x = torch.randn(1, 20, 32, requires_grad=True)

    out, _ = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert not x.grad.isnan().any()
    assert x.grad.norm().item() > 0.0


def test_gradient_flow_causal():
    """Same check under causal=True, since the causal masking path
    (future-candidate substitution in LSHGraphBuilder, causal window
    truncation) is separate code from the non-causal path and could in
    principle break gradients independently."""
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1,
                     causal=True)
    model = SparseTransformer(cfg, num_layers=2)
    x = torch.randn(1, 20, 32, requires_grad=True)

    out, _ = model(x)
    out.sum().backward()

    assert not x.grad.isnan().any()


def test_gradient_flow_cross_attention():
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    attn = SparseAttention(cfg)
    q = torch.randn(1, 10, 32, requires_grad=True)
    kv = torch.randn(1, 20, 32, requires_grad=True)

    out, _ = attn(q, key_value=kv)
    out.sum().backward()

    assert not q.grad.isnan().any()
    assert not kv.grad.isnan().any()


def test_gradient_flow_all_projection_weights_receive_grad():
    """Every learned projection (Wq/Wk/Wv/Wo) should receive a non-zero
    gradient -- a regression here would mean some path through the
    sparse routing is silently detaching part of the computation graph
    (e.g. an .item() or .detach() call accidentally left in a hot path)."""
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    attn = SparseAttention(cfg)
    x = torch.randn(2, 24, 32, requires_grad=True)

    out, _ = attn(x)
    out.sum().backward()

    for name, p in [("Wq", attn.Wq), ("Wk", attn.Wk), ("Wv", attn.Wv), ("Wo", attn.Wo)]:
        assert p.weight.grad is not None, f"{name} received no gradient"
        assert p.weight.grad.norm().item() > 0.0, f"{name} gradient is all zero"
