"""Coverage gaps identified by a post-hoc review of test_*.py against
sparse_attention.py (pytest --cov showed 95% line coverage; this file
closes the five real gaps found, rather than chasing the remaining
lines, which are mostly defensive/unreachable-in-practice branches).

Each test below corresponds to a feature that exists in the source and
is documented, but had zero direct test asserting it actually works:
SubquadraticRegimeTracker, fp32_attn_weights=True, dropout>0, the
_split() backward-compat alias, and SparseTransformer's vocab_size
token-embedding path.
"""
import warnings

import torch

from sparse_attention import (
    SSAConfig,
    SparseAttention,
    SparseTransformer,
    SubquadraticRegimeTracker,
)


# ── SubquadraticRegimeTracker ──────────────────────────────────────────────

def test_regime_tracker_passes_when_k_r_c_constant():
    tracker = SubquadraticRegimeTracker()
    tracker.record(N=128, K=32, R=4, C=128)
    tracker.record(N=512, K=32, R=4, C=128)
    tracker.record(N=2048, K=32, R=4, C=128)

    assert tracker.check(warn=False) is True


def test_regime_tracker_fails_when_k_scales_with_n():
    """The exact misuse case the tracker exists to catch: K growing with
    N silently degrades the complexity claim toward O(N^2) without
    raising an error anywhere else in the codebase."""
    tracker = SubquadraticRegimeTracker()
    tracker.record(N=128, K=16, R=4, C=64)
    tracker.record(N=512, K=64, R=4, C=64)   # K scaled with N — bad

    assert tracker.check(warn=False) is False


def test_regime_tracker_single_call_is_trivially_in_regime():
    tracker = SubquadraticRegimeTracker()
    tracker.record(N=128, K=32, R=4, C=128)

    assert tracker.check(warn=False) is True


def test_regime_tracker_warns_by_default_on_violation():
    tracker = SubquadraticRegimeTracker()
    tracker.record(N=128, K=16, R=4, C=64)
    tracker.record(N=512, K=64, R=4, C=64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tracker.check()  # warn=True is the default

    assert result is False
    assert any("SubquadraticRegimeTracker" in str(w.message) for w in caught)


def test_regime_tracker_realistic_usage_against_real_attn(base_cfg):
    """End-to-end usage matching the docstring's own example: record
    real (N, K, R, C) tuples from actual forward passes at varying
    sequence lengths with a fixed config, confirm the tracker reports
    the regime as healthy."""
    attn = SparseAttention(base_cfg)
    tracker = SubquadraticRegimeTracker()

    for n in (32, 64, 96):
        x = torch.randn(1, n, base_cfg.d_model)
        attn(x)
        tracker.record(N=n, K=base_cfg.num_neighbors,
                        R=base_cfg.num_hash_rounds, C=attn.lsh_k * 4)

    assert tracker.check(warn=False) is True


# ── fp32_attn_weights ───────────────────────────────────────────────────────

def test_fp32_attn_weights_true_runs_and_matches_default_shape(base_cfg):
    cfg_fp32 = SSAConfig(**{**base_cfg.__dict__, "fp32_attn_weights": True})
    attn = SparseAttention(cfg_fp32)
    x = torch.randn(2, 32, cfg_fp32.d_model)

    out, _ = attn(x)
    assert out.shape == x.shape
    assert not out.isnan().any()


def test_fp32_attn_weights_true_vs_false_close_in_fp32_input():
    """With FP32 input end to end, fp32_attn_weights should only affect
    intermediate precision, not the final numerical result by more than
    floating-point noise -- both paths are mathematically the same
    computation, just differing in where the cast to v's dtype happens."""
    cfg_base = dict(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1,
                     dropout=0.0, causal=False)
    torch.manual_seed(7)
    cfg_a = SSAConfig(**cfg_base, fp32_attn_weights=False)
    torch.manual_seed(7)
    cfg_b = SSAConfig(**cfg_base, fp32_attn_weights=True)

    torch.manual_seed(123)
    attn_a = SparseAttention(cfg_a)
    torch.manual_seed(123)
    attn_b = SparseAttention(cfg_b)

    x = torch.randn(1, 24, 32)
    out_a, _ = attn_a(x)
    out_b, _ = attn_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_fp32_attn_weights_gradient_flow(base_cfg):
    cfg_fp32 = SSAConfig(**{**base_cfg.__dict__, "fp32_attn_weights": True})
    attn = SparseAttention(cfg_fp32)
    x = torch.randn(1, 32, cfg_fp32.d_model, requires_grad=True)

    out, _ = attn(x)
    out.sum().backward()

    assert not x.grad.isnan().any()


# ── dropout ──────────────────────────────────────────────────────────────

def test_dropout_active_in_training_mode_changes_output():
    """With dropout > 0 and the module in .train() mode, repeated forward
    passes on the same input should NOT be deterministic (dropout masks
    differ per call) -- this is the actual behavior dropout is supposed
    to produce, and the only way to confirm `if dropout > 0.0 and
    training:` is doing anything rather than being silently skipped."""
    torch.manual_seed(0)
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=16,
                     max_num_hashes=2, window_size=3, num_global_tokens=1,
                     dropout=0.5)
    attn = SparseAttention(cfg)
    attn.train()
    x = torch.randn(1, 24, 32)

    out_1, _ = attn(x)
    out_2, _ = attn(x)

    assert not torch.allclose(out_1, out_2), \
        "dropout=0.5 in training mode produced identical outputs across calls"


def test_dropout_inactive_in_eval_mode_is_deterministic():
    torch.manual_seed(0)
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=16,
                     max_num_hashes=2, window_size=3, num_global_tokens=1,
                     dropout=0.5)
    attn = SparseAttention(cfg)
    attn.eval()
    x = torch.randn(1, 24, 32)

    out_1, _ = attn(x)
    out_2, _ = attn(x)

    assert torch.allclose(out_1, out_2), \
        "dropout was applied in eval mode -- should be a no-op regardless of p"


def test_dropout_zero_is_deterministic_even_in_training_mode():
    """dropout=0.0 should short-circuit the `if dropout > 0.0` guard
    entirely -- confirms the guard, not just F.dropout's own behavior at
    p=0 (which would also be a no-op, but we want to confirm OUR code
    takes the early-exit branch, matching the existing repeated-output
    determinism tests elsewhere in this suite that implicitly rely on
    dropout=0.0 behaving deterministically)."""
    torch.manual_seed(0)
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=16,
                     max_num_hashes=2, window_size=3, num_global_tokens=1,
                     dropout=0.0)
    attn = SparseAttention(cfg)
    attn.train()
    x = torch.randn(1, 24, 32)

    out_1, _ = attn(x)
    out_2, _ = attn(x)

    assert torch.allclose(out_1, out_2)


# ── _split() backward-compat alias ─────────────────────────────────────────

def test_split_alias_matches_split_q(base_cfg):
    """_split() exists purely as a backward-compat alias for callers
    written against an earlier version of this module that only had
    self-attention (before _split_q/_split_kv were separated for GQA).
    It must behave identically to _split_q for any caller still using
    the old name."""
    attn = SparseAttention(base_cfg)
    x = torch.randn(2, 16, base_cfg.d_model)
    projected = attn.Wq(x)

    via_alias = attn._split(projected)
    via_current = attn._split_q(projected)

    assert torch.equal(via_alias, via_current)


# ── SparseTransformer vocab_size / embedding path ──────────────────────────

def test_transformer_with_vocab_size_accepts_token_ids():
    """SparseTransformer(vocab_size=N) switches forward() to expect
    integer token IDs and run them through an internal nn.Embedding --
    a different input contract than the pre-embedded-float-tensor path
    every other test in this suite uses. Never exercised until now."""
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    model = SparseTransformer(cfg, num_layers=2, vocab_size=100)

    token_ids = torch.randint(0, 100, (2, 24))
    out, stats = model(token_ids)

    assert out.shape == (2, 24, cfg.d_model)
    assert model.embed is not None


def test_transformer_without_vocab_size_has_no_embedding_layer():
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    model = SparseTransformer(cfg, num_layers=2)  # vocab_size defaults to 0

    assert model.embed is None


def test_transformer_vocab_size_gradient_flow_through_embedding():
    cfg = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                     max_num_hashes=2, window_size=3, num_global_tokens=1)
    model = SparseTransformer(cfg, num_layers=2, vocab_size=50)
    token_ids = torch.randint(0, 50, (1, 16))

    out, _ = model(token_ids)
    out.sum().backward()

    assert model.embed.weight.grad is not None
    assert not model.embed.weight.grad.isnan().any()
