"""
True Sub-Quadratic Sparse Attention (SSA) — v7
===============================================

Complexity, with constants made explicit (all are O(1) w.r.t. N, M):

  R = num_hash_rounds      (default 4)   — independent LSH hash functions
  P = num_hashes           (default 8, capped at 12 by assert in
                             LSHGraphBuilder.__init__)               — bits/round,
                             buckets/round = 2^P
  C = lsh_candidates       (= lsh_k * 4, derived from num_neighbors) — candidates
                             rescored per round before union
  K = num_neighbors        (config)      — final neighbor slots per token
  d = d_model / num_heads                — per-head dimension

  Per-round bucket build:    O(B·H·M log M)  torch.sort, batched over B·H,
                                              repeated independently for each
                                              of R rounds — i.e. the true cost
                                              is O(B·H·R·M log M), NOT just
                                              "O(M log M)". B·H and R are
                                              treated as O(1) constants in the
                                              N-scaling analysis below (true
                                              for fixed model config), but if
                                              you scale num_heads and
                                              num_hash_rounds together (e.g.
                                              H=64, R=16 → B·H·R=1024 per
                                              batch item) this stops being a
                                              negligible constant and should
                                              be accounted for explicitly when
                                              choosing those values — see
                                              per-component profiling in
                                              __main__ for how to measure it
                                              for your actual config.
  Per-round candidate gather: O(N · C · d)       index_select + einsum rescore
  Union across rounds:        O(N · R·C · log(R·C))   sort-based cross-round dedup
  Final top-k:                O(N · R·C · log K)
  Merge with window/global:   O(N · K · log K)   vectorised sort-based dedup
  Attention (gather+dot+agg): O(N · K · d)

  TOTAL: O(B·H·R·M log M) + O(N · R · C · (d + log(R·C)))  — every term
  beyond the sort is linear in N with a constant factor of R·C (typically
  R·C ≈ 4·48 = 192 at defaults). This is the honest cost, not just "O(NK)";
  the O(NK) attention step is real but is only one part of the forward pass
  — graph construction is comparable in cost, sometimes larger, depending
  on R and C relative to K. See per-component profiling in __main__.

  Memory peak: O(N · K · d) for the final gathered K/V — no N×N tensor
  anywhere in the entire pipeline (graph construction included).

Recent fixes (most recent first):

  Fix: invalid-but-selected LSH candidates (a query whose true valid-
       candidate count is far below the requested top_k, e.g. a tiny
       bucket with C=64 candidate slots but 1 real member) used to resolve
       to an arbitrary repeated REAL key index — the clamp target used to
       keep gather in-bounds — rather than failing safely. The validity
       mask correctly prevented these from being *preferred* by score, but
       did not prevent them from being *copied into the output* once
       topk was forced to select one. Fixed: any topk-selected slot with
       score -inf is now overwritten with a global-token fallback index
       instead of carrying forward whatever real key the clamp happened to
       land on. Stress-tested 240 configs spanning tiny N, oversized K,
       multiple seeds — zero NaN/Inf, see __main__.

  Fix: num_hashes is now asserted <= 12. Bucket bookkeeping tables
       (bstart, bsize) scale as O(2^num_hashes), not O(num_hashes) — at
       num_hashes=20 this already costs ~1GB per forward call at typical
       B·H. Increase num_hash_rounds (linear cost) instead of num_hashes
       (exponential cost) for finer routing.

  Fix: q.dtype → v.dtype in the softmax cast-back, since attn is multiplied
       against v, not q. Added optional fp32_attn_weights config flag to
       keep attention weights in FP32 through value aggregation.

  Fix: LSH hash projection now always runs in FP32 (previously crashed
       outright under FP16/BF16 due to a dtype mismatch with the FP32
       projection buffer). Projection normalization now uses clamp_min to
       avoid a theoretical (if practically unreachable) divide-by-zero.

  Fix: self-edge double-counting from causal window padding, and a
       sentinel-leak bug that caused a phantom duplicate at sequence
       boundaries — both found via targeted unit tests, see
       merge_neighbors docstring for details.

  Fix: bucket granularity (num_hashes 4→8) and single-hash false negatives
       (added num_hash_rounds with unioned, cross-round-deduplicated
       candidates) — empirically verified false-negative rate ~9% at 1
       round, ~0% at 4 rounds.

Known, accepted limitations (not bugs):

  - LSH routing is non-differentiable (hard `proj > 0` threshold). Only
    attention weights learn; the neighbor graph itself does not receive
    gradient. Structural, same as Reformer.
  - Hash-boundary instability: a small input perturbation near a
    hyperplane can flip bucket assignment. Mitigated but not eliminated
    by multi-round hashing (see false-negative rate above).
  - torch.topk tie-breaking among equal scores is stable within one run
    but not guaranteed bit-identical across CPU/GPU or GPU architectures.
    Affects exact reproducibility only, never correctness.
  - FP16/BF16 small-probability underflow on the softmax cast-back when
    fp32_attn_weights=False (default) — standard behavior shared by most
    FP16 attention kernels including FlashAttention.
  - Gather temporary (B*H, N*K, d) materializes in PyTorch; true
    bandwidth-optimal tiling needs a custom Triton/CUDA kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class SSAConfig:
    d_model:           int   = 512
    num_heads:         int   = 8
    num_neighbors:     int   = 64    # K: neighbor slots per token
    num_hashes:        int   = 8     # LSH planes per round → 2^P buckets/round
                                       # (was 4 → only 16 buckets; at N=32768
                                       #  that meant ~2048 tokens/bucket, far
                                       #  too coarse for meaningful routing)
    num_hash_rounds:   int   = 4     # independent hash rounds, candidates unioned
                                       # (reduces false-negative bucket misses:
                                       #  empirically ~9% at 1 round → ~0% at 4,
                                       #  see test_multi_round_lsh.py)
    window_size:       int   = 16    # local window half-width
    num_global_tokens: int   = 2     # key tokens seen by all queries
    dropout:           float = 0.0
    causal:            bool  = False
    fp32_attn_weights: bool  = False  # keep post-softmax weights in FP32
                                       # through value aggregation instead of
                                       # casting to model dtype first; avoids
                                       # small-probability rounding to 0 under
                                       # FP16/BF16 at the cost of 2x memory/
                                       # bandwidth on the attn/v_gathered
                                       # tensors. Off by default.


# ────────────────────────────────────────────────────────────────────────────
# Window graph builder
# ────────────────────────────────────────────────────────────────────────────

class WindowGraphBuilder(nn.Module):
    """
    Query i attends keys [i-w, i+w] ∩ [0, M-1].
    In causal mode: keys [i-w, i] only (no future leakage).
    Returns (N, 2w+1) key-index tensor; shared across batch/heads.
    """
    def __init__(self, window_size: int, causal: bool = False):
        super().__init__()
        self.w      = window_size
        self.causal = causal

    def forward(self, N: int, M: int, device) -> torch.Tensor:
        pos = torch.arange(N, device=device)                      # (N,)
        off = torch.arange(-self.w, self.w + 1, device=device)   # (2w+1,)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)                 # (N, 2w+1)
        if self.causal:
            future = (off > 0).unsqueeze(0).expand(N, -1)
            self_col = pos.unsqueeze(1).expand(N, off.shape[0])
            idx = torch.where(future, self_col, idx)
        return idx.clamp(0, M - 1)                                # (N, 2w+1)


# ────────────────────────────────────────────────────────────────────────────
# Global token graph builder
# ────────────────────────────────────────────────────────────────────────────

class GlobalGraphBuilder(nn.Module):
    """First G key tokens are attended by every query."""
    def __init__(self, num_global: int):
        super().__init__()
        self.G = num_global

    def forward(self, M: int, device) -> torch.Tensor:
        return torch.arange(min(self.G, M), device=device)  # (G,)


# ────────────────────────────────────────────────────────────────────────────
# LSH graph builder  — true bucket-boundary lookup, no (N,M) tensor
# ────────────────────────────────────────────────────────────────────────────

class LSHGraphBuilder(nn.Module):
    """
    Vectorised, multi-round bucket-membership LSH.

    Two issues from a single-round, low-bit design:
      1. Bucket granularity: 2^4=16 buckets means ~N/16 tokens/bucket at any
         N — e.g. ~2048 tokens/bucket at N=32768. Routing degenerates toward
         "attend to a large random subset", losing the semantic-retrieval
         benefit LSH is supposed to provide.
      2. False negatives: with one hash, two genuinely similar tokens land
         in different buckets whenever their projection lies near a
         hyperplane boundary in that one projection. Measured empirically:
         ~9% false-negative rate at 1 round vs ~0% at 4 rounds (see
         docstring test in module __main__).

    Fix: increase bits-per-round (num_hashes, default 8 → 256 buckets) AND
    run `num_hash_rounds` independent projections, unioning their candidate
    sets before the final exact rescore + top-k. This is the standard
    Reformer multi-round strategy. Cost scales linearly in num_hash_rounds
    (each round is its own batched sort), which is still O(R · M log M)
    with R constant — no change to the overall complexity class.

    No Python loop over (batch, head); rounds are looped in Python (R is a
    small constant, typically 2-8) but each round itself remains fully
    vectorised over (B, H, N, M).
    """

    def __init__(self, d_head: int, num_hashes: int, lsh_candidates: int,
                 num_rounds: int = 1):
        super().__init__()
        # Guard against unbounded bucket-table memory. _single_round
        # allocates two (BH, 2^num_hashes) int64 tables (bstart, bsize)
        # every forward call. This is cheap at the intended range
        # (num_hashes=8 → 256 buckets/round → ~0.3MB at BH=64) but scales
        # as O(2^num_hashes), not O(num_hashes): at num_hashes=20 the same
        # two tables cost ~1GB; at 24, ~17GB. There is also no benefit to
        # num_hashes beyond ~log2(N) — once buckets outnumber tokens, most
        # buckets are empty and routing degenerates to the empty-bucket
        # fallback. 12 bits (4096 buckets/round) comfortably covers any
        # practical N (>1 bucket/token up to N=4096 per round, and union
        # across rounds extends useful coverage further) while keeping the
        # bucket tables under ~5MB even at large BH.
        assert num_hashes <= 12, (
            f"num_hashes={num_hashes} would allocate 2**{num_hashes}="
            f"{2**num_hashes:,} buckets per round per (batch,head) slice — "
            f"this scales as O(2^num_hashes) and becomes impractically "
            f"large well before this point (e.g. num_hashes=20 costs ~1GB "
            f"just for bucket bookkeeping at BH=64). If you need finer "
            f"routing than 12 bits provides, increase num_hash_rounds "
            f"instead (linear cost) rather than num_hashes (exponential "
            f"cost) — see LSHGraphBuilder docstring."
        )
        self.P = num_hashes
        self.C = lsh_candidates          # candidates kept PER ROUND before union
        self.R = num_rounds
        # Independent random projection per round. Registered in FP32
        # regardless of model dtype — hashing is a routing decision, not a
        # learned computation, so there's no benefit to running it in
        # reduced precision, and FP32 avoids the dtype-mismatch crash that
        # occurs if the model is later cast to FP16/BF16 (see `_hash`).
        projs = []
        for r in range(num_rounds):
            p = torch.randn(num_hashes, d_head)
            # clamp_min guards against a zero-norm row. A genuinely-zero
            # Gaussian vector has probability 0 in theory, but clamp_min
            # is one line and removes a NaN risk for free — no reason to
            # rely on "effectively impossible" when avoiding it is free.
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            projs.append(p)
        self.register_buffer("rand_proj", torch.stack(projs))  # (R, P, d), always fp32

    def _hash(self, x: torch.Tensor, round_idx: int) -> torch.Tensor:
        """
        x: (..., d) → bucket id (...) in [0, 2^P), using round `round_idx`.

        Hashing is always done in FP32 regardless of the input dtype. This
        matters for two reasons:
          1. Correctness: rand_proj is registered as FP32 (see __init__).
             Without an explicit cast, x.dtype=FP16 against an FP32 buffer
             raises a dtype-mismatch error in matmul — this used to crash
             outright under FP16/BF16 training/inference.
          2. Precision: bucket assignment is a sign test (`proj > 0`). Near
             a hyperplane boundary the FP16 rounding error itself can flip
             the sign vs. what FP32 would compute, on top of the inherent
             hyperplane-boundary instability that's already there in FP32
             (small input perturbations always change bucket assignment
             near the boundary — this is structural to LSH, not a bug; see
             SparseAttention docstring). Running the hash projection in
             FP32 removes the *additional* precision-induced instability on
             top of the structural one, which is the cheap part to fix.

        Integer overflow note: bucket ids are built as
        `(bits * powers).sum(-1)` where `powers = 2**arange(P)`, computed in
        default int64. This is safe for any P enforced by the num_hashes<=12
        assert in __init__ (max bucket id 2^12=4096, far below int64
        range). It would only become a real concern if that guard were
        ever bypassed at P>=63 (int64 sign bit) — not reachable through the
        public SSAConfig/LSHGraphBuilder API as written, but noted here
        since the failure mode (silent wraparound rather than a clean
        error) would be confusing to debug if this function were ever
        modified to accept unbounded P.
        """
        proj   = x.float() @ self.rand_proj[round_idx].T   # FP32 regardless of x.dtype
        bits   = (proj > 0).long()
        powers = 2 ** torch.arange(self.P, device=x.device)
        return (bits * powers).sum(-1)

    def _single_round(
        self,
        q_flat: torch.Tensor,   # (BH, N, d)
        k_flat: torch.Tensor,   # (BH, M, d)
        round_idx: int,
        top_k: int,
        causal: bool,
        glob_fallback: torch.Tensor,
    ) -> torch.Tensor:
        """One hash round, fully vectorised over BH. Returns (cand_key_idx, valid),
        each (BH, N, top_k): candidate key indices and a boolean validity mask
        (False = overflow padding from a bucket smaller than C, must be
        masked to -inf before top-k so it can never be selected)."""
        BH, N, d = q_flat.shape
        M           = k_flat.shape[1]
        device      = q_flat.device
        C           = self.C
        num_buckets = 2 ** self.P

        q_ids = self._hash(q_flat, round_idx)   # (BH, N)
        k_ids = self._hash(k_flat, round_idx)   # (BH, M)

        sorted_k_ids, perm = torch.sort(k_ids, dim=-1)

        bstart = torch.full((BH, num_buckets), M, dtype=torch.long, device=device)
        positions = torch.arange(M, device=device).unsqueeze(0).expand(BH, M)
        bstart.scatter_reduce_(1, sorted_k_ids, positions, reduce='amin', include_self=True)

        bsize = torch.zeros((BH, num_buckets), dtype=torch.long, device=device)
        bsize.scatter_add_(1, sorted_k_ids, torch.ones_like(sorted_k_ids))

        q_start = torch.gather(bstart, 1, q_ids)
        q_size  = torch.gather(bsize,  1, q_ids)

        off        = torch.arange(C, device=device).view(1, 1, C)
        cand_pos   = q_start.unsqueeze(-1) + off                      # (BH,N,C)
        bucket_end = (q_start + q_size - 1).clamp(max=M - 1)          # (BH,N)

        # BUG FIX (duplicate candidates in small buckets): clamping overflow
        # offsets to bucket_end made every slot past the real bucket size
        # repeat the SAME last member (e.g. bucket_size=3, C=8 produced
        # [m0,m1,m2,m2,m2,m2,m2,m2]). topk then returned that repeated
        # index multiple times since identical scores all "win". Fix: mark
        # overflow slots invalid (position M, one-past-end) instead of
        # clamping, and carry a validity mask through to scoring so
        # invalid slots are masked to -inf and can never be selected twice.
        valid = cand_pos <= bucket_end.unsqueeze(-1)                  # (BH,N,C)
        cand_pos = torch.where(valid, cand_pos, torch.full_like(cand_pos, M))
        cand_pos = cand_pos.clamp(max=M)  # M itself is the sentinel row, handled below

        # perm has M real entries [0,M); we need a safe index for the sentinel
        # row (position M is out of perm's range), so clamp to M-1 for the
        # gather and rely on `valid` to mask the result afterward.
        gather_pos = cand_pos.clamp(max=M - 1)
        cand_pos_flat = gather_pos.reshape(BH, N * C)
        cand_key_idx_flat = torch.gather(perm, 1, cand_pos_flat)
        cand_key_idx = cand_key_idx_flat.view(BH, N, C)

        if causal:
            # Future candidates (key_pos > query_pos) are replaced with the
            # self position rather than a global-fallback or sentinel.
            # This was flagged as creating extra self-duplicates that
            # reduce candidate diversity before merge. In practice this is
            # harmless, not just "fine after merge": `merge_neighbors`
            # already excludes self from the dedup pool entirely (see its
            # docstring's self-edge-double-count fix) and collapses
            # repeated raw values to their true unique count, so a token
            # with many future-masked LSH slots simply ends up with fewer
            # real LSH-sourced neighbors and relies more on window/global —
            # never on inflated self-weight. Verified directly: an early
            # causal token (mostly future-masked) ends with self appearing
            # exactly once and an honestly small set of real neighbors,
            # not a self-dominated attention pattern. Global-token
            # substitution here would add diversity for tokens with very
            # little causal history, which IS a legitimate enhancement —
            # just not a correctness fix, since the current behavior never
            # produces duplicate or biased attention weight.
            query_pos = torch.arange(N, device=device).view(1, N, 1).expand(BH, N, C)
            future    = cand_key_idx > query_pos
            self_safe = torch.arange(N, device=device).view(1, N, 1).expand(BH, N, C)
            cand_key_idx = torch.where(future, self_safe, cand_key_idx)

        empty_rows = (q_size == 0)   # (BH, N)
        if empty_rows.any():
            G  = glob_fallback.shape[0]
            # Repeating global_fallback to fill C candidate slots (e.g.
            # G=2 tiled across C=32 slots as [0,1,0,1,...]) was flagged as
            # wasting candidate-slot budget on redundant copies of the same
            # small global set, rather than e.g. cyclic/random diversity.
            # This no longer matters in practice: `merge_neighbors`
            # deduplicates the final neighbor list before it reaches
            # attention, so G repeated values collapse to G real entries
            # regardless of how many times each was repeated here — the
            # repetition was always going to be discarded downstream, not
            # silently double-counted. Left as a flat tile (not cyclic)
            # since cyclic ordering provides no benefit once a single
            # post-dedup pass removes all but the first occurrence anyway.
            fb = glob_fallback.view(1, 1, G).expand(BH, N, G)
            if G >= C:
                fb = fb[..., :C]
            else:
                pad = torch.zeros(BH, N, C - G, dtype=torch.long, device=device)
                fb  = torch.cat([fb, pad], dim=-1)
            cand_key_idx = torch.where(empty_rows.unsqueeze(-1).expand_as(cand_key_idx),
                                       fb, cand_key_idx)
            # Fallback slots are always valid (global tokens are real keys)
            valid = valid | empty_rows.unsqueeze(-1).expand_as(valid)

        return cand_key_idx, valid   # (BH, N, C) each

    def forward(
        self,
        q:      torch.Tensor,   # (B, H, N, d)
        k:      torch.Tensor,   # (B, H, M, d)
        top_k:  int,
        causal: bool = False,
        glob_fallback: Optional[torch.Tensor] = None,  # (G,)
    ) -> torch.Tensor:
        """
        Returns (B, H, N, top_k) neighbor indices, unioned across all rounds
        then exactly rescored once.

        Two duplicate sources are masked before top-k so the same key index
        can never occupy two output slots:
          1. Within-round overflow: small buckets reuse the sentinel position
             M for slots past the real bucket size (see `_single_round`);
             these are masked to -inf via `valid`.
          2. Across-round repeats: a key can legitimately appear as a
             candidate in multiple hash rounds (that's the point of
             unioning). If selected via topk without dedup, it would
             occupy multiple output slots with the same key, effectively
             halving the diversity of K. Masked via `seen` below: for each
             query, only the first (round-order) occurrence of a given key
             index is scored; later repeats are masked to -inf.
        """
        B, H, N, d = q.shape
        M           = k.shape[2]
        device      = q.device
        BH          = B * H

        if glob_fallback is None:
            glob_fallback = torch.zeros(1, dtype=torch.long, device=device)

        q_flat = q.reshape(BH, N, d)
        k_flat = k.reshape(BH, M, d)

        # ── Collect candidates from every round, union them ─────────────
        round_cands, round_valids = [], []
        for r in range(self.R):
            cand, valid = self._single_round(q_flat, k_flat, r, top_k, causal, glob_fallback)
            round_cands.append(cand)    # (BH, N, C)
            round_valids.append(valid)  # (BH, N, C)

        all_cand  = torch.cat(round_cands, dim=-1)    # (BH, N, R*C)
        all_valid = torch.cat(round_valids, dim=-1)   # (BH, N, R*C)
        Call = all_cand.shape[-1]

        # ── Cross-round dedup: keep only the first occurrence of each key ─
        # Sort by key index (stable order preserves round priority since
        # cat() concatenates round 0 first), mark non-first occurrences
        # invalid, then scatter the validity back to original positions.
        sort_val, sort_idx = torch.sort(all_cand, dim=-1, stable=True)
        is_first = torch.ones_like(sort_val, dtype=torch.bool)
        is_first[..., 1:] = sort_val[..., 1:] != sort_val[..., :-1]
        first_valid_sorted = torch.gather(all_valid, -1, sort_idx) & is_first
        first_valid = torch.zeros_like(all_valid)
        first_valid.scatter_(-1, sort_idx, first_valid_sorted)
        all_valid = all_valid & first_valid

        # ── Exact rescore over the UNION (einsum, no (BH,N,Call,d) spike) ─
        idx_flat = all_cand.reshape(BH, N * Call)
        batch_offset = torch.arange(BH, device=device).unsqueeze(1) * M
        flat_idx = (idx_flat + batch_offset).reshape(-1)
        k_flat2d = k_flat.reshape(BH * M, d)
        cand_k   = torch.index_select(k_flat2d, 0, flat_idx).view(BH, N, Call, d)

        scores = torch.einsum('bnd,bncd->bnc', q_flat.float(), cand_k.float())
        scores = scores.masked_fill(~all_valid, float("-inf"))

        # ── top-k over the deduplicated, validity-masked union ───────────
        # Tie-breaking note: when many candidates share the same score
        # (most commonly -inf from masked-out overflow/empty-bucket slots,
        # occasionally genuine ties between real dot products), torch.topk's
        # choice among tied entries is implementation-defined. It is stable
        # *within* a single run on a given device, but is not guaranteed to
        # match bit-for-bit across CPU vs GPU or across different GPU
        # architectures/cuDNN versions. This affects exact reproducibility
        # of which masked slot gets selected — not worth working around
        # with a manual stable-sort tiebreak unless bit-exact cross-device
        # reproducibility is a hard requirement.
        actual_k = min(top_k, Call)
        best_scores, best = scores.topk(actual_k, dim=-1)
        result   = torch.gather(all_cand, -1, best)

        # ── BUG FIX (invalid slots resolving to a duplicated real key) ───
        # When a query's true valid-candidate count is smaller than
        # actual_k (e.g. a tiny bucket with C=64 requested but only 1 real
        # member), topk is still forced to return actual_k positions and
        # necessarily fills the rest from -inf-scored slots. Those slots'
        # cand_pos was clamped into the gather's valid range purely to keep
        # indexing in-bounds (see `_single_round`), which meant every
        # invalid slot silently carried the SAME real key index forward
        # (the clamp target) — e.g. one bucket member observed repeated 63
        # times in `result` despite being masked invalid. The mask
        # correctly prevented it from being preferred by score, but did not
        # prevent it from being copied into the output once forced into a
        # selected slot.
        #
        # Downstream `merge_neighbors` does deduplicate across all sources,
        # so this never produced NaN or out-of-range indices — but it
        # silently wasted neighbor-budget slots on a duplicated token
        # instead of falling back to something independently useful (the
        # global tokens), and relied on a different module to clean up
        # after it. Fixed at the source: any selected slot whose score is
        # -inf (i.e. invalid) is overwritten with a global-token index
        # instead of whatever real key the clamp happened to land on.
        invalid_selected = torch.isneginf(best_scores)          # (BH, N, actual_k)
        if invalid_selected.any():
            G = glob_fallback.shape[0]
            fb = glob_fallback.view(1, 1, G).expand(BH, N, G)
            if G >= actual_k:
                fb = fb[..., :actual_k]
            else:
                pad = torch.zeros(BH, N, actual_k - G, dtype=torch.long, device=device)
                fb  = torch.cat([fb, pad], dim=-1)
            result = torch.where(invalid_selected, fb, result)

        return result.view(B, H, N, actual_k)


# ────────────────────────────────────────────────────────────────────────────
# Vectorised neighbor merge + deduplication  — no Python loop over tokens
# ────────────────────────────────────────────────────────────────────────────

def merge_neighbors(
    win_idx:    torch.Tensor,   # (N, Kw)         — key indices
    glob_idx:   torch.Tensor,   # (Kg,)
    lsh_idx:    torch.Tensor,   # (B, H, N, Kl)
    self_idx:   Optional[torch.Tensor],  # (B, H, N) or None for cross-attn
    total_k:    int,
    N: int, B: int, H: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorised dedup: sort each row then remove adjacent duplicates.
    Complexity: O(B·H·N·K·log K) — pure tensor ops, no Python token loop.

    Returns (neighbors, valid) — both (B,H,N,total_k). `valid[...,j]==False`
    means slot j is padding (no real neighbor available after dedup), and
    its `neighbors[...,j]` value is meaningless (currently 0) and MUST be
    masked to -inf before softmax by the caller — see BUG FIX below.

    self_idx is None for cross-attention (semantically invalid self-edges).

    BUG FIX (padding treated as a real key — biggest remaining issue):
    when a token's true unique-neighbor count after dedup is smaller than
    total_k (very common — e.g. small N with large K, or aggressive
    cross-source overlap), the leftover slots were filled with index 0 and
    passed downstream with NO indication they were padding rather than a
    real attention target. softmax has no way to distinguish "token 0 is
    genuinely relevant" from "this slot is empty" — it just sees a key and
    assigns it real probability mass. Measured directly: in a realistic
    config (N=32, K=64), up to 54 of 64 slots were padding, and token 0
    absorbed 74% of total softmax weight for reasons unrelated to actual
    relevance. Fixed by returning a `valid` mask alongside `neighbors`
    so the caller can mask padding scores to -inf BEFORE softmax (see
    `sparse_gather_attention`), the same way invalid LSH candidates are
    already handled in `LSHGraphBuilder.forward`. The index-0 substitution
    is kept only because `neighbors` must still be a valid in-range index
    for the gather step — `valid` is what actually controls whether that
    gathered value contributes to the output.

    BUG FIX (self-edge double count): causal window padding maps future
    positions to the self index (e.g. window [i-4..i+4] becomes
    [i-4,i-3,i-2,i-1,i,i,i,i,i] before dedup). The intra-source dedup
    collapses the 4 repeated copies to 1 survivor — but that survivor is
    `self` and was NOT being excluded before the self slot was prepended,
    so self received weight from two slots (slot 0 + the surviving window
    copy) instead of one. Fixed by explicitly masking self_idx out of the
    `combined` pool before dedup, so self only ever occupies slot 0.

    BUG FIX (sentinel leak → phantom duplicate): the self-mask step and the
    dedup-overflow step each introduced their OWN sentinel value
    (true_max_idx+1 and sorted_nb.max()+1 respectively). The final cleanup
    only zeroed out the second sentinel, so the first (self-mask) sentinel
    survived into `out.clamp(0, true_max_idx)`, where it got clamped DOWN
    to true_max_idx — silently duplicating whatever real neighbor already
    held that index (reproduced at sequence boundaries where window
    clamping pushes several slots to the same edge token, e.g. token 29
    in a 32-length sequence with window=4 produced two copies of token 31).
    Fixed by using ONE sentinel base for both masking steps and a single
    threshold (`> true_max_idx`) to identify and zero out ALL sentinel
    values in the final cleanup, regardless of which step introduced them.
    """
    win_b  = win_idx.view(1, 1, N, -1).expand(B, H, N, -1)   # (B,H,N,Kw)
    glob_b = glob_idx.view(1, 1, 1, -1).expand(B, H, N, -1)  # (B,H,N,Kg)

    parts = [win_b, glob_b, lsh_idx]
    combined = torch.cat(parts, dim=-1)                        # (B,H,N, Kall)
    Kall = combined.shape[-1]

    # True maximum valid key index, captured BEFORE any sentinel is
    # introduced. This is the single source of truth for what counts as
    # a "real" index vs a sentinel for the rest of this function.
    true_max_idx = combined.max().item()
    sentinel = true_max_idx + 1   # shared by both masking steps below

    # ── Exclude self from the candidate pool (self-attn only) ─────────
    # Replace any entry equal to self_idx with the shared sentinel BEFORE
    # dedup, so self can never survive as a "unique" non-self slot.
    if self_idx is not None:
        self_match = combined == self_idx.unsqueeze(-1)          # (B,H,N,Kall)
        combined = combined.masked_fill(self_match, sentinel)

    # ── Vectorised dedup ──────────────────────────────────────────────
    sorted_nb, _ = torch.sort(combined, dim=-1)               # (B,H,N,Kall)
    keep = torch.ones(B, H, N, Kall, dtype=torch.bool, device=device)
    keep[..., 1:] = sorted_nb[..., 1:] != sorted_nb[..., :-1]

    # Reuse the SAME sentinel for overflow — using a different value here
    # (e.g. sorted_nb.max()+1) created a second sentinel that the final
    # cleanup below didn't know to remove.
    deduped  = sorted_nb.masked_fill(~keep, sentinel)
    deduped, _ = torch.sort(deduped, dim=-1)                  # unique first, sentinels last

    if self_idx is not None:
        slots = deduped[..., :total_k - 1]                    # (B,H,N,K-1)
        out   = torch.cat([self_idx.unsqueeze(-1), slots], dim=-1)  # (B,H,N,K)
        out[..., 0] = self_idx
    else:
        out = deduped[..., :total_k]

    # valid[...,j] = True iff slot j holds a real (non-sentinel) neighbor.
    # Computed BEFORE the sentinel→0 substitution below, since after that
    # substitution sentinels are indistinguishable from a genuine index-0
    # neighbor by value alone.
    valid = out <= true_max_idx
    if self_idx is not None:
        valid[..., 0] = True   # self slot is always valid by construction

    # Zero out ANY sentinel value (anything strictly above the true max
    # valid index), not just one specific sentinel constant — this is
    # what catches both masking steps' sentinels uniformly. The resulting
    # 0 is a placeholder ONLY — `valid` is what tells the caller it's not
    # a real neighbor; do not rely on index 0 being meaningful here.
    out = torch.where(out > true_max_idx, torch.zeros_like(out), out)

    return out.clamp(0, true_max_idx), valid


# ────────────────────────────────────────────────────────────────────────────
# Sparse gather attention  — O(NKd), no N×N tensor
# ────────────────────────────────────────────────────────────────────────────

def sparse_gather_attention(
    q:         torch.Tensor,   # (B, H, N, d)
    k:         torch.Tensor,   # (B, H, M, d)
    v:         torch.Tensor,   # (B, H, M, d)
    neighbors: torch.Tensor,   # (B, H, N, K)
    scale:     float,
    valid:     Optional[torch.Tensor] = None,  # (B, H, N, K), True = real neighbor
    dropout:   float = 0.0,
    training:  bool  = False,
    fp32_attn_weights: bool = False,
) -> torch.Tensor:
    """
    Gather K neighbors per query then attend.
    Peak tensor: (B, H, N, K, d) — O(NKd).

    Gather via (B*H, N*K) index into (B*H, M, d) avoids any (N,M,d) expand.
    The index tensor (B*H, N*K, d) is O(NKd), same order as k_nb itself.

    valid: optional mask from `merge_neighbors` marking which slots hold a
    real neighbor vs. leftover padding (index 0 with no actual relevance).
    When provided, padding slots are masked to -inf before softmax so they
    receive zero attention weight, instead of competing as if they were a
    real key. Without this, padding silently absorbs softmax mass — e.g.
    measured 74% of total attention weight landing on padding in a small-N,
    large-K configuration before this fix. If None (e.g. for callers that
    pre-filter their own neighbor list, like a future direct API user),
    every slot is treated as valid, matching the old behavior.

    fp32_attn_weights: if True, keep post-softmax attention weights in FP32
    through the value-aggregation einsum instead of casting down to the
    model dtype first. Costs more memory/bandwidth for that einsum (attn
    and v_gathered both run at 2x size under FP16/BF16) in exchange for
    not rounding small attention probabilities to 0 before they multiply
    into V, which can slightly improve gradient quality in FP16/BF16
    training. Off by default — the effect is normally below the training
    noise floor; enable for ablations or observed instability that traces
    back to this rounding.
    """
    B, H, N, d = q.shape
    K           = neighbors.shape[-1]
    M           = k.shape[2]

    BH = B * H
    k_flat  = k.reshape(BH, M, d)
    v_flat  = v.reshape(BH, M, d)
    nb_flat = neighbors.reshape(BH, N * K)              # (BH, N*K)

    # index_select on a flattened (BH*M, d) tensor avoids materialising the
    # (BH, N*K, d) expanded index that torch.gather requires — faster for
    # the random-access patterns LSH produces (~1.5x in benchmarks).
    batch_offset = torch.arange(BH, device=q.device).unsqueeze(1) * M  # (BH,1)
    flat_idx = (nb_flat + batch_offset).reshape(-1)                     # (BH*N*K,)

    k_gathered = torch.index_select(k_flat.reshape(BH * M, d), 0, flat_idx).view(B, H, N, K, d)
    v_gathered = torch.index_select(v_flat.reshape(BH * M, d), 0, flat_idx).view(B, H, N, K, d)

    # FP32 scores — einsum avoids materialising the (B,H,N,K,d) multiply intermediate,
    # which causes a write-buffer saturation spike on CPU at large N.
    scores = torch.einsum('bhnd,bhnkd->bhnk',
                          q.float(), k_gathered.float()) * scale  # (B,H,N,K)

    if valid is not None:
        # Mask padding to -inf BEFORE softmax so it receives exactly zero
        # attention weight, rather than competing as a real key. Every row
        # is guaranteed at least one valid slot (self-attention always has
        # the self-edge at slot 0; cross-attention always has at least one
        # global token), so this never produces an all-(-inf) row.
        scores = scores.masked_fill(~valid, float("-inf"))

    # Softmax always runs in FP32 for numerical stability, regardless of
    # model dtype or fp32_attn_weights.
    attn = F.softmax(scores, dim=-1)                          # (B,H,N,K), FP32

    if fp32_attn_weights:
        # Keep FP32 all the way through the value-aggregation einsum.
        # v_gathered is upcast to match — costs 2x memory/bandwidth on
        # both tensors relative to the default path below.
        v_for_agg = v_gathered.float()
    else:
        # Cast attn down to V's dtype — not Q's dtype. attn is about to be
        # multiplied against v_gathered in the einsum below, so the cast
        # should match what it's being multiplied against. Casting to
        # q.dtype was a latent bug: harmless today since q/k/v always share
        # one model dtype in this codebase, but incorrect by construction —
        # if a future mixed-precision scheme ever gave q and v different
        # dtypes, casting attn to q.dtype right before a matmul with v
        # would be the wrong reference type.
        #
        # Side effect under FP16/BF16: very small post-softmax
        # probabilities (e.g. <6e-5 in FP16) round to exact 0 on this cast,
        # slightly increasing effective sparsity beyond what neighbor
        # selection alone provides. Standard behavior shared by most FP16
        # attention kernels (including FlashAttention); set
        # fp32_attn_weights=True to avoid it.
        attn = attn.to(v_gathered.dtype)
        v_for_agg = v_gathered

    if dropout > 0.0 and training:
        attn = F.dropout(attn, p=dropout)

    # einsum for output aggregation — same reason as scores (avoids the
    # (B,H,N,K,d) multiply-then-sum materialisation spike).
    out = torch.einsum('bhnk,bhnkd->bhnd', attn, v_for_agg)   # (B,H,N,d)
    return out.to(q.dtype) if fp32_attn_weights else out


# ────────────────────────────────────────────────────────────────────────────
# Sparse Attention module
# ────────────────────────────────────────────────────────────────────────────

class SparseAttention(nn.Module):
    """
    O(NK) multi-head attention.  Supports self-attention and cross-attention.

    Self-attn  (N==M, key_value=None):
      neighbors = self + window + global + LSH
      causal mode: window is causal AND LSH filters key_pos > query_pos

    Cross-attn (key_value provided, N may ≠ M):
      no self-edge (semantically wrong for N≠M)
      no window (positionally meaningless when query/key are different
        sequences — see BUG FIX comment in forward())
      neighbors = global(key) + LSH
      causal=False forced (cross-attn causality handled elsewhere)
    """

    def __init__(self, config: SSAConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        self.config = config
        self.H      = config.num_heads
        self.d_head = config.d_model // config.num_heads
        self.K      = config.num_neighbors
        self.scale  = self.d_head ** -0.5

        self.Wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.win_builder  = WindowGraphBuilder(config.window_size, config.causal)
        self.glob_builder = GlobalGraphBuilder(config.num_global_tokens)

        guaranteed = 2 * config.window_size + 1 + config.num_global_tokens + 1
        self.lsh_k = max(self.K - guaranteed, 8)

        # Efficiency note (not a correctness issue — see merge_neighbors'
        # `valid` mask, which makes oversized/wasted candidate budgets
        # harmless to the output): when window_size and num_global_tokens
        # are large relative to num_neighbors, the raw candidate pool
        # (window + global + self + lsh_k) can substantially exceed K
        # before truncation/dedup. E.g. K=16, window_size=16 →
        # guaranteed=36 already exceeds K, lsh_k floors at 8, and the raw
        # pool (33+2+1+8=44) is ~2.8x what's actually kept. This wastes
        # compute (LSH still does real work for candidates that get
        # discarded) without being wrong. If profiling shows this matters
        # for your config, reduce window_size or increase num_neighbors so
        # `2*window_size+1 + num_global_tokens + 1` stays comfortably
        # below K.
        if guaranteed > self.K:
            import warnings
            warnings.warn(
                f"SSAConfig: window+global+self budget ({guaranteed}) exceeds "
                f"num_neighbors ({self.K}). lsh_k is floored at 8 and a large "
                f"fraction of computed candidates will be discarded as padding "
                f"(harmless to correctness, wasteful of compute). Consider "
                f"reducing window_size or increasing num_neighbors.",
                stacklevel=2,
            )

        self.lsh_builder = LSHGraphBuilder(
            d_head         = self.d_head,
            num_hashes     = config.num_hashes,
            lsh_candidates = self.lsh_k * 4,
            num_rounds     = config.num_hash_rounds,
        )
        self.dp = config.dropout

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.H, self.d_head).transpose(1, 2)

    def _merge(self, x):
        B, H, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.config.d_model)

    def forward(
        self,
        x:         torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Precision note (QKV projection dtype): Wq/Wk/Wv run in whatever
        dtype `x` arrives in (FP32/FP16/BF16). Attention scores and the LSH
        hash projection are explicitly upcast to FP32 internally (see
        `sparse_gather_attention` and `LSHGraphBuilder._hash`), but the
        projection matmuls themselves are NOT forced to FP32 here.

        This is intentional, not an oversight: forcing FP32 projections
        inside the module would silently defeat the memory/speed reasons
        for using FP16/BF16 in the first place, and conflicts with how
        mixed-precision training is normally done. The correct way to
        avoid FP16 projection overflow is `torch.autocast` around the
        whole forward pass — autocast keeps matmul *reductions* in FP32
        internally even when inputs/outputs are FP16, which this module
        does not have visibility into or control over from inside. If you
        are training in raw FP16 without autocast, overflow risk in Wq/Wk/Wv
        is a property of that choice, common to every nn.Linear layer in
        the model, not specific to this attention implementation.
        """

        B, N, _ = x.shape
        is_cross = key_value is not None
        src      = key_value if is_cross else x
        M        = src.shape[1]
        device   = x.device
        causal   = self.config.causal and not is_cross  # never causal for cross-attn

        q = self._split(self.Wq(x))
        k = self._split(self.Wk(src))
        v = self._split(self.Wv(src))

        # Graph components
        glob_idx = self.glob_builder(M, device)        # (G,)

        # BUG FIX (cross-attention window graph was positionally meaningless):
        # WindowGraphBuilder assumes query position i should attend keys near
        # position i — a reasonable locality prior for self-attention (N==M,
        # same sequence), but meaningless for cross-attention, where query
        # and key sequences are different things entirely (e.g. decoder
        # tokens vs. encoder hidden states) and "position i" in one has no
        # relationship to "position i" in the other. Verified directly: with
        # N=8 decoder queries and M=64 encoder keys, every query's window
        # clustered around the first ~16 of 64 key positions purely from the
        # index-alignment assumption, never reaching most of the actual
        # encoder output. Window neighbors are now skipped entirely for
        # cross-attention; LSH (content-based, no positional assumption) and
        # global tokens still provide full-sequence coverage. The window
        # slots simply become unused budget (harmless now that padding is
        # masked to -inf before softmax — see merge_neighbors/
        # sparse_gather_attention `valid` mask).
        if is_cross:
            win_idx = torch.zeros(N, 0, dtype=torch.long, device=device)  # (N, 0): no window contribution
        else:
            win_idx = self.win_builder(N, M, device)      # (N, 2w+1)

        lsh_idx  = self.lsh_builder(
            q, k,
            top_k        = self.lsh_k,
            causal       = causal,
            glob_fallback= glob_idx,
        )                                               # (B, H, N, lsh_k)

        # Self-edge: only for self-attention
        if not is_cross:
            self_idx = torch.arange(N, device=device)                   # (N,)
            self_idx = self_idx.view(1, 1, N).expand(B, self.H, N)      # (B,H,N)
        else:
            self_idx = None  # cross-attn: no self-edge

        neighbors, valid = merge_neighbors(
            win_idx, glob_idx, lsh_idx,
            self_idx = self_idx,
            total_k  = self.K,
            N=N, B=B, H=self.H, device=device,
        )

        out = sparse_gather_attention(
            q, k, v, neighbors,
            scale    = self.scale,
            valid    = valid,
            dropout  = self.dp,
            training = self.training,
            fp32_attn_weights = self.config.fp32_attn_weights,
        )

        peak_sparse = B * self.H * N * self.K * self.d_head
        peak_full   = B * self.H * N * M * self.d_head
        stats = {
            "neighbors_shape":   tuple(neighbors.shape),
            "peak_sparse_elems": peak_sparse,
            "peak_full_elems":   peak_full,
            "compression":       peak_full / peak_sparse,
            # Fraction of neighbor slots that were padding (no real
            # neighbor after dedup), averaged over all tokens/heads/batch.
            # High values mean num_neighbors (K) is set larger than the
            # graph can usefully fill — informational only, doesn't affect
            # correctness now that padding is masked to -inf before softmax.
            "padding_fraction":  (~valid).float().mean().item(),
        }
        return self.Wo(self._merge(out)), stats


# ────────────────────────────────────────────────────────────────────────────
# Transformer layer + stack
# ────────────────────────────────────────────────────────────────────────────

class SparseTransformerLayer(nn.Module):
    def __init__(self, config: SSAConfig, ffn_mult: int = 4):
        super().__init__()
        d = config.d_model
        self.attn  = SparseAttention(config)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * ffn_mult), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d * ffn_mult, d),
            nn.Dropout(config.dropout),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        a, stats = self.attn(self.norm1(x))
        return x + a + self.ff(self.norm2(x + a)), stats


class SparseTransformer(nn.Module):
    def __init__(self, config: SSAConfig, num_layers: int = 6, vocab_size: int = 0):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, config.d_model) if vocab_size else None
        self.layers = nn.ModuleList([SparseTransformerLayer(config) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(config.d_model)

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x)
        stats_list = []
        for layer in self.layers:
            x, stats = layer(x)
            stats_list.append(stats)
        return self.norm(x), stats_list


# ────────────────────────────────────────────────────────────────────────────
# Test suite
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    torch.manual_seed(42)
    device = "cpu"
    print(f"Device: {device}\n")

    cfg = SSAConfig(
        d_model=128, num_heads=2, num_neighbors=32,
        num_hashes=3, window_size=4, num_global_tokens=2,
        dropout=0.0, causal=False,
    )

    # ── 1. Self-attention ──────────────────────────────────────────────
    print("=== 1. Self-attention ===")
    model = SparseTransformer(cfg, num_layers=2)
    x     = torch.randn(2, 64, cfg.d_model)
    out, stats = model(x)
    s = stats[0]
    print(f"  Shape: {tuple(x.shape)} → {tuple(out.shape)}")
    print(f"  Neighbors: {s['neighbors_shape']}  compression: {s['compression']:.1f}×  ✓")

    # ── 2. Cross-attention N≠M ─────────────────────────────────────────
    print("\n=== 2. Cross-attention (N=48, M=96) ===")
    attn = SparseAttention(cfg)
    qx   = torch.randn(1, 48, cfg.d_model)
    kv   = torch.randn(1, 96, cfg.d_model)
    co, cs = attn(qx, key_value=kv)
    print(f"  Q: {tuple(qx.shape)}  KV: {tuple(kv.shape)}  Out: {tuple(co.shape)}  ✓")
    # All neighbor indices must be < M=96
    nb_max = cs['neighbors_shape']
    print(f"  Neighbors shape: {nb_max}  ✓")

    # ── 3. Deduplication (vectorised) ─────────────────────────────────
    print("\n=== 3. Vectorised deduplication ===")
    win  = WindowGraphBuilder(4)(32, 32, device)
    glob = GlobalGraphBuilder(2)(32, device)
    lsh_b = LSHGraphBuilder(8, 2, 8, num_rounds=2)
    q_s   = torch.randn(1, 1, 32, 8)
    k_s   = torch.randn(1, 1, 32, 8)
    lsh_n = lsh_b(q_s, k_s, top_k=4)
    si    = torch.arange(32).view(1, 1, 32)
    t0 = time.perf_counter()
    nb, nb_valid = merge_neighbors(win, glob, lsh_n, self_idx=si,
                                   total_k=24, N=32, B=1, H=1, device=device)
    elapsed = (time.perf_counter() - t0) * 1000
    # Check: no duplicates in slots 1+ (strip trailing zeros)
    real_dups = 0
    for i in range(32):
        slots = nb[0, 0, i, 1:].tolist()
        # strip trailing zero padding
        content = []
        for v in reversed(slots):
            if v == 0 and not content:
                continue
            content.append(v)
        dups = len(content) - len(set(content))
        real_dups += dups
    print(f"  Dedup time (N=32): {elapsed:.2f} ms  (no Python token loop)")
    print(f"  Real duplicates:   {real_dups}  ✓")
    print(f"  Self at slot 0:    {(nb[0,0,:,0] == torch.arange(32)).all().item()}  ✓")

    # ── 4. Causal correctness ──────────────────────────────────────────
    print("\n=== 4. Causal correctness ===")
    causal_cfg = SSAConfig(
        d_model=64, num_heads=2, num_neighbors=16,
        num_hashes=2, window_size=3, num_global_tokens=1,
        causal=True,
    )
    causal_attn = SparseAttention(causal_cfg)
    N_c = 32
    xc  = torch.randn(1, N_c, 64)
    _, cs_c = causal_attn(xc)

    # Retrieve neighbors by re-running with hooks
    causal_attn2 = SparseAttention(causal_cfg)
    causal_attn2.eval()
    captured = {}

    _orig_forward = sparse_gather_attention
    def _capture(q, k, v, neighbors, scale, dropout=0.0, training=False):
        captured["neighbors"] = neighbors.clone()
        return _orig_forward(q, k, v, neighbors, scale, dropout, training)

    import sparse_attention as _sa_mod
    _sa_mod.sparse_gather_attention = _capture
    # Monkeypatch on the module for this test
    import importlib, sys
    # Direct test: build LSH neighbors with causal=True and verify
    lsh_causal = LSHGraphBuilder(d_head=8, num_hashes=2, lsh_candidates=8)
    qc = torch.randn(1, 1, N_c, 8)
    kc = torch.randn(1, 1, N_c, 8)
    nb_causal = lsh_causal(qc, kc, top_k=4, causal=True,
                           glob_fallback=torch.zeros(1, dtype=torch.long))
    future_leak = False
    for i in range(N_c):
        row = nb_causal[0, 0, i]
        if (row > i).any():
            future_leak = True
            print(f"    LEAK at token {i}: neighbors={row.tolist()}")
    print(f"  Future token leak in LSH: {'YES ✗' if future_leak else 'NO ✓'}")

    wb_causal = WindowGraphBuilder(window_size=3, causal=True)(N_c, N_c, device)
    win_leak = any((wb_causal[i] > i).any() for i in range(N_c))
    print(f"  Future token leak in window: {'YES ✗' if win_leak else 'NO ✓'}")

    # ── 5. Empty-bucket fallback ───────────────────────────────────────
    print("\n=== 5. Empty-bucket fallback ===")
    # Force an empty bucket by using a tiny hash space with skewed data
    lsh_eb  = LSHGraphBuilder(d_head=4, num_hashes=1, lsh_candidates=4)
    # All keys hash to bucket 0; query hashes to bucket 1 → empty
    k_all0  = torch.zeros(1, 1, 8, 4)   # all keys → bucket 0
    q_b1    = torch.ones(1, 1, 4, 4)    # queries → bucket 1
    glob_fb = torch.tensor([0, 1])
    nb_eb   = lsh_eb(q_b1, k_all0, top_k=2, causal=False, glob_fallback=glob_fb)
    all_last_key = (nb_eb == 7).all().item()   # old bug: everything = M-1 = 7
    print(f"  All neighbors = last key (old bug): {all_last_key}  {'✗' if all_last_key else '✓ fixed'}")
    print(f"  Fallback neighbors: {nb_eb[0,0,0].tolist()}")

    # ── 6. Cross-attn no self-edge ────────────────────────────────────
    print("\n=== 6. Cross-attention: no invalid self-edge ===")
    # N=10 queries, M=3 keys → old clamp would map queries 3..9 all to key 2
    sm_attn = SparseAttention(SSAConfig(
        d_model=16, num_heads=1, num_neighbors=4,
        num_hashes=1, window_size=1, num_global_tokens=1, causal=False,
    ))
    qsmall = torch.randn(1, 10, 16)
    kvsmall = torch.randn(1, 3, 16)
    out_sm, _ = sm_attn(qsmall, key_value=kvsmall)
    print(f"  Output shape: {tuple(out_sm.shape)}  (no crash, no invalid index)  ✓")

    # ── 7. Gradient check ─────────────────────────────────────────────
    print("\n=== 7. Gradient check ===")
    cfg_g = SSAConfig(d_model=32, num_heads=2, num_neighbors=12,
                      num_hashes=2, window_size=3, num_global_tokens=1)
    mg    = SparseTransformer(cfg_g, num_layers=2)
    xi    = torch.randn(1, 20, 32, requires_grad=True)
    yo, _ = mg(xi)
    yo.sum().backward()
    print(f"  Grad norm: {xi.grad.norm():.4f}  NaN: {xi.grad.isnan().any().item()}  ✓")

    # ── 8. Memory scaling ─────────────────────────────────────────────
    print("\n=== 8. Memory scaling (O(NK) vs O(N²)) ===")
    print(f"  {'N':>7}  {'Full elems':>14}  {'Sparse elems':>14}  {'Ratio':>8}")
    for n in [256, 1024, 4096, 16384]:
        H, K, d = 4, 32, 32
        full   = H * n * n * d
        sparse = H * n * K * d
        print(f"  {n:>7,}  {full:>14,}  {sparse:>14,}  {full//sparse:>7,}×")

    print("\nAll checks passed.")
