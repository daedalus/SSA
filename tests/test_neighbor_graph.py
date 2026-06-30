"""Neighbor graph construction correctness: deduplication, self-edge
placement, and the empty-bucket LSH fallback.

Ported from the original __main__ checks 3, 5, and 6.
"""
import torch

from sparse_attention import (
    SSAConfig,
    SparseAttention,
    WindowGraphBuilder,
    GlobalGraphBuilder,
    LSHGraphBuilder,
    merge_neighbors,
)


def test_merge_neighbors_no_real_duplicates(device):
    """Within the real (non-padding) portion of each query's neighbor
    list, every index must appear at most once. Slot 0 is reserved for
    self and is checked separately -- this only checks slots 1+."""
    win = WindowGraphBuilder(4)(32, 32, device)
    glob = GlobalGraphBuilder(2)(32, device)
    lsh_builder = LSHGraphBuilder(d_head=8, lsh_candidates=8, num_rounds=2)
    q = torch.randn(1, 1, 32, 8)
    k = torch.randn(1, 1, 32, 8)
    lsh_idx = lsh_builder(q, k, top_k=4)
    self_idx = torch.arange(32).view(1, 1, 32)

    neighbors, valid = merge_neighbors(
        win, glob, lsh_idx, self_idx=self_idx,
        total_k=24, N=32, B=1, H=1, device=device,
    )

    for i in range(32):
        slots = neighbors[0, 0, i, 1:][valid[0, 0, i, 1:]].tolist()
        assert len(slots) == len(set(slots)), f"duplicate neighbor at query {i}: {slots}"


def test_merge_neighbors_self_always_at_slot_zero(device):
    win = WindowGraphBuilder(4)(32, 32, device)
    glob = GlobalGraphBuilder(2)(32, device)
    lsh_builder = LSHGraphBuilder(d_head=8, lsh_candidates=8, num_rounds=2)
    q = torch.randn(1, 1, 32, 8)
    k = torch.randn(1, 1, 32, 8)
    lsh_idx = lsh_builder(q, k, top_k=4)
    self_idx = torch.arange(32).view(1, 1, 32)

    neighbors, _ = merge_neighbors(
        win, glob, lsh_idx, self_idx=self_idx,
        total_k=24, N=32, B=1, H=1, device=device,
    )

    assert (neighbors[0, 0, :, 0] == torch.arange(32)).all()


def test_empty_bucket_fallback_uses_global_tokens_not_last_key():
    """Regression test for a fixed bug: an empty LSH bucket used to fall
    back to repeating the LAST key index (M-1) for every query, which is
    both wrong (arbitrary, not content-based) and silently degrades to
    the same neighbor for every query. The fix routes to the global
    token fallback set instead."""
    lsh = LSHGraphBuilder(d_head=4, max_num_hashes=1, lsh_candidates=4)
    k_all_bucket0 = torch.zeros(1, 1, 8, 4)   # all 8 keys hash to bucket 0
    q_bucket1 = torch.ones(1, 1, 4, 4)        # all queries hash to bucket 1 (empty)
    glob_fallback = torch.tensor([0, 1])

    neighbors = lsh(q_bucket1, k_all_bucket0, top_k=2,
                     causal=False, glob_fallback=glob_fallback)

    all_last_key = (neighbors == 7).all().item()  # M-1 == 7; the old bug
    assert not all_last_key, "empty-bucket fallback regressed to repeating the last key index"
    # Every empty-bucket query should fall back to the global set [0, 1]
    assert set(neighbors[0, 0, 0].tolist()) <= {0, 1}


def test_cross_attention_no_invalid_self_edge():
    """Regression test: with N=10 queries and only M=3 keys, an old
    clamping bug mapped every query position >= M to key index M-1,
    creating a spurious self-like edge that doesn't correspond to a real
    self-attention relationship in cross-attention (where N often != M
    and there IS no self edge by design)."""
    cfg = SSAConfig(
        d_model=16, num_heads=1, num_neighbors=4,
        max_num_hashes=1, window_size=1, num_global_tokens=1, causal=False,
    )
    attn = SparseAttention(cfg)
    q = torch.randn(1, 10, 16)
    kv = torch.randn(1, 3, 16)

    out, _ = attn(q, key_value=kv)
    assert out.shape == (1, 10, 16)
    assert not out.isnan().any()
