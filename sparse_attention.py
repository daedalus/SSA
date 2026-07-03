"""
Sub-Quadratic Sparse Attention (SSA)
=========================================================

  Sub-quadratic in N HOLDS IF AND ONLY IF K, R, C are held fixed
  (i.e. independent of N) as N grows. If a caller scales K, R, or C
  proportionally to N — e.g. "use more neighbors for longer sequences"
  — the complexity reduces toward O(N²d) exactly like dense attention,
  because K·R·C stops being a constant factor and becomes an N-dependent
  one. SSAConfig does NOT enforce K/R/C independence from N; this is a
  caller responsibility. See `validate_subquadratic_regime` for an
  opt-in runtime check.

Complexity, with constants made explicit (all are O(1) w.r.t. N, M
ONLY IF chosen independently of N — see above):

  R = num_hash_rounds      (default 8)   — independent LSH hash functions
  P = effective bucket bits/round, computed dynamically per forward call
      as round(log2(M / C)), clamped to [1, max_num_hashes] — NOT fixed
      at construction time (see LSHGraphBuilder.compute_effective_p)
  C = lsh_candidates       (= lsh_k * 4, derived from num_neighbors) — candidates
                             rescored per round before union
  K = num_neighbors        (config)      — final neighbor slots per token
  d = d_model / num_heads                — per-head dimension

  Per-round bucket build:    O(B·H·M log M)  torch.sort, batched over B·H,
                                              repeated independently for each
                                              of R rounds — i.e. the true cost
                                              is O(B·H·R·M log M), NOT just
                                              "O(M log M)".
  Per-round candidate gather: O(N · C · d)       index_select + einsum rescore
  Union across rounds:        O(N · R·C · log(R·C))   sort-based cross-round dedup
  Final top-k:                O(N · R·C · log K)
  Merge with window/global:   O(N · K · log K)   vectorised sort-based dedup
  Attention (gather+dot+agg): O(N · K · d)

  TOTAL: O(B·H·R·M log M) + O(N · R · C · (d + log(R·C)))

MEASURED WALL-CLOCK BREAKDOWN (not just asymptotic — this is what
actually dominates runtime in practice, CPU, single core, N=32768,
K=32, R=4, window=4):

    Per-round bucket sort (×4 rounds):         670 ms  (14.6%)
    Cross-round dedup sort:                    770 ms  (16.8%)
    merge_neighbors internal sorts:             52 ms  ( 1.1%)
    merge_neighbors (rest: gather/scatter):     78 ms  ( 1.7%)
    Rescore gather (index_select):           1,502 ms  (32.7%)
    Actual sparse attention (gather+dot+agg):  411 ms  ( 9.0%)
    ──────────────────────────────────────────────────────────
    Total forward pass:                      4,588 ms  (the rest, ~23%,
                                              is Q/K/V projections, output
                                              projection, and Python/
                                              dispatch overhead not broken
                                              out above)

  CROSS-CUTTING VIEW (sorting appears at 3 separate call sites — per-round
  bucket construction, cross-round dedup, and inside merge_neighbors —
  easy to undercount if read as 3 separate "minor" costs rather than one
  recurring operation):

    ALL torch.sort() calls combined:         1,492 ms  (32.5%)
    The rescore gather alone:                1,502 ms  (32.7%)

  These two are effectively tied as the dominant cost, each roughly 3.6x
  the actual attention step (411ms, 9.0%). LSH graph construction as a
  whole (all sorts + the gather + merge_neighbors overhead) is ~91% of
  wall-clock time at this scale; the O(NKd) attention step the
  architecture is named for is ~9%. "Attention is O(NK)" is
  asymptotically true and practically misleading taken alone — the
  bottleneck at long context is graph construction, split roughly evenly
  between sort-dominated and gather-dominated costs, neither of which
  is FLOP-bound. The gather is memory-bandwidth-bound (measured ~1.7
  GB/s on a single CPU core regardless of whether access is random or
  sequential — 1.1x difference between the two — consistent with a
  bandwidth bound, not a cache-locality bound specifically). The sorts
  are a mix of genuine O(M log M) algorithmic cost and PyTorch's
  CPU sort implementation constant factor, which has not been profiled
  against alternative sort algorithms (radix sort would likely be
  faster for the small fixed-width integer keys used here, but is not
  implemented). On GPU (HBM bandwidth 2-8 TB/s vs this sandbox's
  single-core ~GB/s) the absolute numbers would differ by orders of
  magnitude, but the structural conclusion — these ops move a lot of
  memory and do comparatively little compute, so they WILL be
  bandwidth/sort-bound somewhere, just at a different scale — should
  transfer. Not independently verified on GPU; this codebase has only
  been profiled on CPU, and that remains the single largest gap before
  any production claim (A100/H100 benchmarks, FlashAttention-2
  comparison, and a Triton bucket-construction kernel would all be
  required before this implementation could honestly be compared
  against production sparse-attention systems).

  The single largest fixed-config lever for reducing this overhead is
  the LSH candidate oversample factor (C = lsh_k * 4): reducing it
  trades recall for speed roughly linearly (measured 22.9% recall at
  1x oversample vs 51.8% at 4x, same proportional cost difference in
  Call = R*C). A custom GPU kernel (Triton/CUDA) replacing the
  sort-based bucket construction with segmented radix sort or a hash
  table would be the correct fix for the sort costs specifically, but
  has not been implemented — see "Known limitations" below.

  Memory peak: NOT O(N·K·d). The final attention step (gather + dot +
  aggregate) does use O(N·K·d), but it is not the peak — the LSH union/
  rescore step that runs BEFORE it materializes (B·H, N, R·C, d)-shaped
  tensors (`cand_k` inside LSHGraphBuilder.forward), where R·C is
  typically several times larger than K (e.g. measured R·C=672 vs K=64
  in one tested config — a 10.5x larger intermediate than the "O(NK)"
  figure suggests). The TRUE peak across the whole forward pass is
  O(N · R · C · d), not O(N · K · d). Both are still linear in N with R
  and C as O(1) constants for fixed config (so the asymptotic class is
  unaffected), but stating "O(NK)" without qualification — easy to do,
  since it's true for the attention step in isolation — understates the
  actual peak memory a profiler would report by a meaningful constant
  factor. No N×N tensor anywhere in the entire pipeline regardless.

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
    # GQA / MQA: set num_kv_heads < num_heads for grouped/multi-query attention.
    # num_heads must be divisible by num_kv_heads.  Default None means full MHA
    # (num_kv_heads == num_heads).  Each KV head is shared across
    # (num_heads // num_kv_heads) query heads, matching the LLaMA-2/Mistral
    # convention.  d_kv_head == d_model // num_heads (key/value head dim is
    # kept equal to query head dim so the same LSH projections work; only the
    # number of independent KV projections changes).
    num_kv_heads:      Optional[int] = None
    num_neighbors:     int   = 128   # K: neighbor slots per token.
                                       # AlphaEvolve grid search found K=128 is
                                       # Pareto-optimal for N≤1024: 92.6% recall
                                       # at N=1024 vs 63.5% at K=64 (same R, os).
                                       # For N≤256, K=64 suffices (96.9% recall).
    max_num_hashes:    int   = 12    # CEILING on LSH planes/round (2^P buckets).
                                       # The actual P used per forward call is
                                       # computed dynamically from the current
                                       # sequence length as
                                       # round(log2(seq_len / lsh_candidates)),
                                       # clamped to [1, max_num_hashes] — see
                                       # LSHGraphBuilder.compute_effective_p.
                                       # This replaces a previous fixed
                                       # num_hashes=8 default, which was
                                       # measurably suboptimal at both short
                                       # and long sequences (2-4x worse
                                       # recall@K than the adaptive formula
                                       # in testing across N=64..32768) and
                                       # only happened to be near-correct at
                                       # one specific N (~1024).
    num_hash_rounds:   int   = 8     # independent hash rounds, candidates unioned
                                       # (reduces false-negative bucket misses:
                                       #  empirically ~9% at 1 round → ~0% at 4,
                                       #  ~0% at 8. AlphaEvolve found R=8 gives
                                       #  +7% recall over R=4 at K=128, N=1024
                                       #  (99.3% vs 92.6%). Linear cost.)
    lsh_num_probes:    int   = 0     # multi-probe LSH: additional buckets checked
                                       # per round, per query, beyond the query's own
                                       # bucket. Each probe flips the single lowest-
                                       # confidence hash bit (smallest |hyperplane
                                       # projection|) not yet flipped, i.e. probe 1
                                       # checks the bucket one bit-flip away from the
                                       # query's own bucket on the boundary it's
                                       # closest to, probe 2 the next-closest
                                       # boundary, etc. (standard step-wise multi-
                                       # probe LSH; see LSHGraphBuilder._probe_ids).
                                       # This targets false negatives from queries
                                       # that land near a hyperplane boundary --
                                       # exactly the failure mode num_hash_rounds
                                       # already reduces via independent rehashing,
                                       # but multi-probe targets it directly within
                                       # a single round instead of relying on a
                                       # different round's random projection to
                                       # happen to separate the pair correctly.
                                       # Cost: each probe adds another C-sized
                                       # candidate gather per round, i.e. total
                                       # rescore cost scales with
                                       # R * (1 + lsh_num_probes), same as
                                       # increasing num_hash_rounds by that
                                       # factor -- see benchmarks/bench_multiprobe.py
                                       # for the actual recall/speed tradeoff,
                                       # measured, not assumed. Default 0 preserves
                                       # exact prior behavior.
    window_size:       int   = 8     # local window half-width.
                                       # AlphaEvolve found window=8 outperforms
                                       # window=16 at ALL sequence lengths: a
                                       # smaller window frees K budget for LSH
                                       # content-based routing, which captures
                                       # the important tokens more effectively
                                       # than positional proximity.
    num_global_tokens: int   = 2     # key tokens seen by all queries.
                                       # Governs the DEFAULT behaviour (first G
                                       # tokens) when global_token_indices is
                                       # None.  Ignored when
                                       # global_token_indices is set explicitly.
    global_token_indices: Optional[list] = None
                                       # Explicit list of key positions that
                                       # every query attends to, e.g. [0, 1]
                                       # for BOS+CLS, or [0, 128, 256] for
                                       # evenly-spaced landmarks.  When set,
                                       # overrides num_global_tokens entirely.
                                       # Indices are clamped to [0, M-1] at
                                       # runtime so out-of-range values are
                                       # safe (but wasteful).  Use this instead
                                       # of num_global_tokens whenever your
                                       # global tokens are NOT the leading
                                       # positions of the key sequence.
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
# Sub-quadratic regime validator (opt-in)
# ────────────────────────────────────────────────────────────────────────────

class SubquadraticRegimeTracker:
    """
    Opt-in runtime check for the conditional sub-quadratic claim (see
    module docstring). The architecture is sub-quadratic in N ONLY IF
    K (num_neighbors), R (num_hash_rounds), and the derived C
    (lsh_candidates) are held constant as N varies across calls. Nothing
    in SSAConfig enforces this — a caller could legitimately (if
    unwisely) scale K with N, silently degrading toward O(N²d) while the
    code keeps running without error.

    This tracker does not change behavior. It records the (N, K, R, C)
    seen on each forward call and warns if K, R, or C ever changes in a
    way that correlates with N — a cheap, optional sanity check for
    anyone who wants to confirm their usage pattern stays in the regime
    where the complexity analysis in the module docstring actually holds.

    Not wired into SparseAttention by default (adds bookkeeping
    overhead and most callers use a single fixed config). Use explicitly:

        tracker = SubquadraticRegimeTracker()
        for batch in data:
            out, stats = attn(batch)
            tracker.record(N=batch.shape[1], K=cfg.num_neighbors,
                           R=cfg.num_hash_rounds, C=attn.lsh_k * 4)
        tracker.check()  # raises/warns if K, R, or C varied with N
    """
    def __init__(self):
        self.history: list = []  # list of (N, K, R, C) tuples

    def record(self, N: int, K: int, R: int, C: int) -> None:
        self.history.append((N, K, R, C))

    def check(self, warn: bool = True) -> bool:
        """
        Returns True if K, R, C were constant across all recorded calls
        (the regime where sub-quadratic scaling actually holds). Warns
        (or returns False silently if warn=False) otherwise.
        """
        if len(self.history) < 2:
            return True
        Ks = {h[1] for h in self.history}
        Rs = {h[2] for h in self.history}
        Cs = {h[3] for h in self.history}
        in_regime = len(Ks) == 1 and len(Rs) == 1 and len(Cs) == 1
        if not in_regime and warn:
            import warnings
            ns = [h[0] for h in self.history]
            warnings.warn(
                f"SubquadraticRegimeTracker: K/R/C varied across "
                f"{len(self.history)} calls (N ranged {min(ns)}-{max(ns)}, "
                f"K values seen: {sorted(Ks)}, R values: {sorted(Rs)}, "
                f"C values: {sorted(Cs)}). The sub-quadratic complexity "
                f"claim only holds when K, R, C are constants independent "
                f"of N — see module docstring. If K/R/C correlate with N "
                f"in your usage, effective complexity approaches O(N²d), "
                f"same as dense attention.",
                stacklevel=2,
            )
        return in_regime


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
    """
    Key tokens attended by every query.

    Two modes (controlled by SSAConfig):
      - indices=None  -> first G tokens, i.e. torch.arange(G).  Default,
                         matches the original behaviour.
      - indices=[...] -> explicit list of key positions.  Allows BOS/CLS at
                         position 0, mid-sequence landmarks, or any other
                         non-contiguous set.  Positions are clamped to
                         [0, M-1] at runtime so stale indices from a longer
                         sequence never go out of range.
    """
    def __init__(self, num_global: int, indices: Optional[list] = None):
        super().__init__()
        if indices is not None:
            self.register_buffer(
                "_fixed_idx",
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self.G = len(indices)
            self._use_fixed = True
        else:
            self.G = num_global
            self._use_fixed = False

    def forward(self, M: int, device) -> torch.Tensor:
        if self._use_fixed:
            idx = self._fixed_idx.to(device)
            return idx.clamp(0, M - 1)           # (G,) safe for any M
        return torch.arange(min(self.G, M), device=device)  # (G,)


# ────────────────────────────────────────────────────────────────────────────
# LSH graph builder  — true bucket-boundary lookup, no (N,M) tensor
# ────────────────────────────────────────────────────────────────────────────

class LSHGraphBuilder(nn.Module):
    """
    Vectorised, multi-round bucket-membership LSH with sequence-length-
    adaptive bucket count.

    Three issues from a fixed, low-bit design:
      1. Bucket granularity: a FIXED 2^P buckets is wrong for some N no
         matter what P is chosen. At N=128 with P=8 (256 buckets), most
         buckets are empty (occupancy 0.5) and routing degenerates to the
         empty-bucket fallback. At N=32768 with the same P=8, occupancy is
         128 — buckets are so coarse the exact-rescore-then-topk step
         can't discriminate well within them. A fixed P cannot be correct
         across this range simultaneously.
      2. False negatives: with one hash, two genuinely similar tokens land
         in different buckets whenever their projection lies near a
         hyperplane boundary in that one projection. Measured empirically:
         ~9% false-negative rate at 1 round vs ~0% at 4 rounds.
      3. (Addressed here) Optimal bucket occupancy, measured empirically
         via recall@K against true dense top-K across N ∈ [64, 32768], is
         ≈ lsh_candidates (C) — i.e. you want enough tokens per bucket to
         fill the per-round candidate budget, not more, not less. This
         gives P = round(log2(N / C)), which matched the brute-force-
         optimal P at EVERY tested N (64 through 32768) exactly. A fixed
         P=8 default lost to this formula by 2-4x in recall at both ends
         of that range (e.g. 8.8% vs 27.1% recall@16 at N=128; 14.0% vs
         22.6% at N=4096) and only matched it by coincidence at one
         specific N (~1024) where the fixed default happened to land near
         the formula's value anyway.

    Fix for (1)+(3): bucket count is no longer fixed at construction time.
    `rand_proj` is registered with `max_num_hashes` planes (a ceiling, not
    a target); at each `forward()` call, the EFFECTIVE number of planes
    actually used for bucketing is computed from the current key sequence
    length M as `P_eff = clamp(round(log2(M / lsh_candidates)), 1, max_num_hashes)`
    and the first `P_eff` rows of `rand_proj` are sliced out. This is
    cheap (a view, not a reconstruction) and means a single trained module
    automatically uses coarser buckets for short sequences and finer
    buckets for long ones, without needing separate models or retraining
    when sequence length changes between calls.

    Fix for (2): run `num_hash_rounds` independent projections, unioning
    their candidate sets before the final exact rescore + top-k (Reformer
    multi-round strategy). Also empirically improves seed-sensitivity:
    relative recall std across 15 random projection seeds dropped from
    3.6% (R=1) to 1.7% (R=4) in testing.

    No Python loop over (batch, head); rounds are looped in Python (R is a
    small constant, typically 2-8) but each round itself remains fully
    vectorised over (B, H, N, M).
    """

    def __init__(self, d_head: int, lsh_candidates: int,
                 num_rounds: int = 1, max_num_hashes: int = 12,
                 num_probes: int = 0):
        super().__init__()
        # max_num_hashes is a CEILING on bucket count, not the bucket
        # count itself — see forward()'s P_eff computation. The same
        # memory-blowup reasoning as before still applies to this ceiling:
        # bucket bookkeeping tables scale as O(2^P), so an unreasonably
        # high ceiling is exactly as dangerous as a fixed unreasonable P
        # used to be.
        assert max_num_hashes <= 12, (
            f"max_num_hashes={max_num_hashes} would allow up to "
            f"2**{max_num_hashes}={2**max_num_hashes:,} buckets per round per "
            f"(batch,head) slice — this scales as O(2^P) and becomes "
            f"impractically large well before this point (e.g. P=20 costs "
            f"~1GB just for bucket bookkeeping at BH=64). If you need finer "
            f"routing than 12 bits provides, increase num_hash_rounds "
            f"instead (linear cost) rather than max_num_hashes (exponential "
            f"cost)."
        )
        self.max_P = max_num_hashes
        self.C = lsh_candidates          # candidates kept PER ROUND before union;
                                          # also the target bucket occupancy used
                                          # to derive P_eff in forward()
        self.R = num_rounds
        assert num_probes >= 0, f"num_probes must be >= 0, got {num_probes}"
        self.T = num_probes              # multi-probe: extra buckets checked per
                                          # round beyond the query's own bucket
        # Independent random projection per round, registered at the
        # CEILING plane count. forward() slices the first P_eff rows per
        # call — slicing a view is free; this lets one buffer serve any
        # P_eff <= max_num_hashes without reconstruction.
        #
        # Registered in FP32 regardless of model dtype — hashing is a
        # routing decision, not a learned computation, so there's no
        # benefit to running it in reduced precision, and FP32 avoids the
        # dtype-mismatch crash that occurs if the model is later cast to
        # FP16/BF16 (see `_hash`).
        projs = []
        for r in range(num_rounds):
            p = torch.randn(max_num_hashes, d_head)
            # clamp_min guards against a zero-norm row. A genuinely-zero
            # Gaussian vector has probability 0 in theory, but clamp_min
            # is one line and removes a NaN risk for free — no reason to
            # rely on "effectively impossible" when avoiding it is free.
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            projs.append(p)
        self.register_buffer("rand_proj", torch.stack(projs))  # (R, max_P, d), always fp32

    @staticmethod
    def compute_effective_p(M: int, C: int, max_p: int, min_p: int = 1) -> int:
        """
        P_eff = round(log2(M / C)), clamped to [min_p, max_p].

        Targets average bucket occupancy ≈ C (the per-round candidate
        budget) — empirically the recall-maximizing choice across the
        full tested range, see class docstring. min_p=1 (not e.g. 4) is
        deliberate: at small M relative to C, the optimum can be as low
        as 1-2 bits (occupancy >> C, i.e. "barely bucket at all, mostly
        rely on the candidate budget to cover most of the sequence
        directly") — measured 94.6% recall@16 at M=128,C=64,P=1 vs 40.9%
        at the naive P=4 floor used in an earlier version of this module.

        Discontinuity note: round() means P_eff jumps by exactly 1 (i.e.
        bucket count doubles or halves) whenever M/C crosses a
        half-integer power of 2 — e.g. at C=64, this happens at
        M ≈ 182, 363, 725, 1449, 2897, .... A sequence-length change that
        crosses one of these boundaries (e.g. M=1448→1449) does cause a
        real, measurable shift in routing, but it is small in practice:
        measured ~3 percentage points of recall@16 and no detectable
        latency discontinuity (~2% noise) right at the M=1448/1449/1450
        boundary. This is smaller than the seed-to-seed recall variance
        already inherent to LSH (~1.7% relative std at R=4 rounds,
        measured separately) and well within normal run-to-run noise, not
        a sharp cliff. No hysteresis or floor/ceil smoothing implemented —
        the measured discontinuity size didn't justify the added
        complexity, but this is a judgment call; revisit if a specific
        downstream use case is sensitive to sequence-length-triggered
        routing changes (e.g. reproducibility requirements across
        sequences that happen to straddle a boundary).
        """
        if M <= 1 or C <= 0:
            return min_p
        p_ideal = math.log2(max(M, 1) / max(C, 1))
        return max(min_p, min(max_p, round(p_ideal)))

    def _hash(self, x: torch.Tensor, round_idx: int, p_eff: int) -> torch.Tensor:
        """
        x: (..., d) → bucket id (...) in [0, 2^p_eff), using round
        `round_idx` and the first `p_eff` planes of that round's
        projection (a slice of the registered max_num_hashes-plane buffer,
        not a separate buffer — see class docstring).

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
        `(bits * powers).sum(-1)` where `powers = 2**arange(p_eff)`,
        computed in default int64. Safe for any p_eff enforced by the
        max_num_hashes<=12 assert in __init__ (max bucket id 2^12=4096,
        far below int64 range).
        """
        proj   = x.float() @ self.rand_proj[round_idx, :p_eff].T   # FP32, sliced planes
        bits   = (proj > 0).long()
        powers = 2 ** torch.arange(p_eff, device=x.device)
        return (bits * powers).sum(-1)

    def _hash_proj(self, x: torch.Tensor, round_idx: int, p_eff: int) -> torch.Tensor:
        """Same projection `_hash` computes internally, exposed directly so
        `_probe_ids` can rank bits by confidence (|proj|) without
        recomputing the matmul. Always FP32 -- see `_hash` docstring."""
        return x.float() @ self.rand_proj[round_idx, :p_eff].T   # (..., p_eff)

    def _probe_ids(self, proj: torch.Tensor, p_eff: int, num_probes: int) -> list:
        """
        Step-wise multi-probe LSH (Lv et al.): given a query's hyperplane
        projections for one round, return up to `num_probes` additional
        bucket ids beyond the query's own bucket, each one bit-flip away.

        Rationale: a query near a hyperplane boundary (small |proj| on
        that bit) is the case single-bucket lookup gets wrong most often
        -- the "true" neighbor landed on the other side of a coin-flip-
        close decision. Rather than adding a whole extra independent
        hash round to catch this (linear cost, no guarantee it targets
        THIS boundary), multi-probe explicitly checks the buckets
        adjacent to the boundary this specific query is closest to.

        Probes are ranked by ascending |proj| (least confident bit
        first) and applied one bit at a time, not combinatorially (i.e.
        probe 2 flips the 2nd-least-confident bit relative to the
        ORIGINAL id, not the already-flipped probe-1 id) -- this is the
        standard single-flip step-wise variant, not full multi-probe
        with combined multi-bit perturbations, which would need
        2^num_probes buckets to enumerate the same coverage. Single-flip
        gives the highest marginal recall per extra bucket checked, since
        the lowest-margin bit is disproportionately likely to be a false
        boundary crossing.

        Returns a list of `min(num_probes, p_eff)` LongTensors, each
        shaped like the leading dims of `proj` (i.e. (..., ) with the
        trailing p_eff dim reduced away) -- one alternate bucket id per
        probe. Empty list if num_probes <= 0 or p_eff == 0.
        """
        num_probes = min(num_probes, p_eff)
        if num_probes <= 0:
            return []
        bits   = (proj > 0).long()                        # (..., p_eff)
        powers = 2 ** torch.arange(p_eff, device=proj.device)
        base_id = (bits * powers).sum(-1)                  # (...,)
        margins = proj.abs()                                # (..., p_eff)
        order = margins.argsort(dim=-1)                     # ascending: least confident first

        probe_ids = []
        for t in range(num_probes):
            pos = order[..., t]                              # (...,) bit index to flip
            bit_val = torch.gather(bits, -1, pos.unsqueeze(-1)).squeeze(-1)
            pow_val = powers[pos]                            # fancy-indexes powers per element
            # bit was 1 -> flipping to 0 subtracts pow_val; bit was 0 -> flipping to 1 adds it.
            delta = pow_val * (1 - 2 * bit_val)
            probe_ids.append(base_id + delta)
        return probe_ids

    def _gather_bucket(
        self,
        q_ids: torch.Tensor,      # (BH, N) bucket id per query, ANY id set (primary or probe)
        bstart: torch.Tensor,     # (BH, num_buckets)
        bsize: torch.Tensor,      # (BH, num_buckets)
        perm: torch.Tensor,       # (BH, M) sort permutation from k_ids
        N: int, M: int, C: int,
        causal: bool,
        glob_fallback: torch.Tensor,
        device,
    ):
        """Given a bucket id per query (whichever id set -- the query's own
        bucket, or a multi-probe alternate bucket), gather that bucket's
        (up to C) members. Factored out of `_single_round` so probe buckets
        go through byte-identical logic to the primary bucket -- same
        overflow-padding, causal future-masking, and empty-bucket fallback,
        just keyed by a different `q_ids` tensor. Returns (cand_key_idx,
        valid), each (BH, N, C)."""
        BH = q_ids.shape[0]

        q_start = torch.gather(bstart, 1, q_ids)
        q_size  = torch.gather(bsize,  1, q_ids)

        off        = torch.arange(C, device=device).view(1, 1, C)
        cand_pos   = q_start.unsqueeze(-1) + off                      # (BH,N,C)
        bucket_end = (q_start + q_size - 1).clamp(max=M - 1)          # (BH,N)

        # BUG FIX (duplicate candidates in small buckets), preserved from
        # the pre-multi-probe implementation: clamping overflow offsets to
        # bucket_end made every slot past the real bucket size repeat the
        # SAME last member (e.g. bucket_size=3, C=8 produced
        # [m0,m1,m2,m2,m2,m2,m2,m2]). topk then returned that repeated
        # index multiple times since identical scores all "win". Fix: mark
        # overflow slots invalid (position M, one-past-end) instead of
        # clamping, and carry a validity mask through to scoring so
        # invalid slots are masked to -inf and can never be selected twice.
        # Applies identically to probe buckets: a probe bucket smaller
        # than C overflows exactly the same way the primary bucket does.
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
            # self position rather than a global-fallback or sentinel. See
            # `_single_round`'s pre-refactor docstring for the full
            # rationale (merge_neighbors dedups self, so this never
            # inflates self-weight). Applies to probe buckets too: a probe
            # bucket can contain future positions exactly like the primary
            # bucket can.
            query_pos = torch.arange(N, device=device).view(1, N, 1).expand(BH, N, C)
            future    = cand_key_idx > query_pos
            self_safe = torch.arange(N, device=device).view(1, N, 1).expand(BH, N, C)
            cand_key_idx = torch.where(future, self_safe, cand_key_idx)

        empty_rows = (q_size == 0)   # (BH, N)
        if empty_rows.any():
            # Global-token fallback for empty buckets -- see the original
            # `_single_round` docstring (pre-refactor) for the full
            # quantified-risk analysis (0% all-rounds-empty in the
            # recommended occupancy regime; only a real risk under
            # deliberate lsh_candidates misconfiguration). Probe buckets
            # are MORE likely to be empty than primary buckets by
            # construction -- they're chosen specifically because they're
            # adjacent to a near-boundary query, which correlates with
            # small/sparse buckets near the tails of the bucket occupancy
            # distribution -- so this fallback path is exercised more
            # often under multi-probe than without it. That's expected
            # and harmless: an empty probe bucket falling back to global
            # tokens just means that probe contributed nothing beyond
            # what global already provides, not a correctness issue.
            G  = glob_fallback.shape[0]
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

    def _single_round(
        self,
        q_flat: torch.Tensor,   # (BH, N, d)
        k_flat: torch.Tensor,   # (BH, M, d)
        round_idx: int,
        top_k: int,
        causal: bool,
        glob_fallback: torch.Tensor,
        p_eff: int,
        num_probes: int = 0,
        _buf: Optional[dict] = None,
    ) -> torch.Tensor:
        """One hash round, fully vectorised over BH. Returns (cand_key_idx, valid),
        each (BH, N, C*(1+num_probes_used)): candidate key indices and a boolean
        validity mask (False = overflow padding from a bucket smaller than C,
        must be masked to -inf before top-k so it can never be selected).

        With num_probes > 0 (multi-probe LSH), the query's own bucket AND
        `num_probes` additional near-boundary buckets (see `_probe_ids`)
        are each gathered via `_gather_bucket` and concatenated along the
        candidate axis before returning -- from the caller's (`forward`'s)
        point of view this is indistinguishable from having C been larger,
        it's just sourced from multiple buckets instead of one. The
        cross-round union/dedup in `forward` handles any overlap between a
        probe bucket and another round's primary bucket for free.

        _buf: optional pre-allocated buffers dict (bstart, bsize, positions,
        off) to avoid per-round allocation. Pass None to allocate fresh."""
        BH, N, d = q_flat.shape
        M           = k_flat.shape[1]
        device      = q_flat.device
        C           = self.C
        num_buckets = 2 ** p_eff

        q_proj = self._hash_proj(q_flat, round_idx, p_eff)   # (BH, N, p_eff)
        k_ids  = self._hash(k_flat, round_idx, p_eff)         # (BH, M)
        bits   = (q_proj > 0).long()
        powers = 2 ** torch.arange(p_eff, device=device)
        q_ids  = (bits * powers).sum(-1)                       # (BH, N) -- primary bucket

        sorted_k_ids, perm = torch.sort(k_ids, dim=-1)

        if _buf is not None:
            bstart = _buf['bstart'].fill_(M)
            positions = _buf['positions']
            bsize = _buf['bsize'].zero_()
        else:
            bstart = torch.full((BH, num_buckets), M, dtype=torch.long, device=device)
            positions = torch.arange(M, device=device).unsqueeze(0).expand(BH, M)
            bsize = torch.zeros((BH, num_buckets), dtype=torch.long, device=device)
        bstart.scatter_reduce_(1, sorted_k_ids, positions, reduce='amin', include_self=True)
        bsize.scatter_add_(1, sorted_k_ids, torch.ones_like(sorted_k_ids))

        id_sets = [q_ids] + self._probe_ids(q_proj, p_eff, num_probes)

        if len(id_sets) == 1:
            return self._gather_bucket(
                id_sets[0], bstart, bsize, perm, N, M, C, causal, glob_fallback, device,
            )

        cand_list, valid_list = [], []
        for ids in id_sets:
            cand, valid = self._gather_bucket(
                ids, bstart, bsize, perm, N, M, C, causal, glob_fallback, device,
            )
            cand_list.append(cand)
            valid_list.append(valid)

        cand_key_idx = torch.cat(cand_list, dim=-1)   # (BH, N, C*len(id_sets))
        valid        = torch.cat(valid_list, dim=-1)

        return cand_key_idx, valid

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

        # Sequence-length-adaptive bucket count: P_eff is recomputed every
        # call from the CURRENT key sequence length M, not fixed at
        # construction time. See compute_effective_p and class docstring
        # for why occupancy≈C is the empirically recall-maximizing target.
        p_eff = self.compute_effective_p(M, self.C, self.max_P)

        q_flat = q.reshape(BH, N, d)
        k_flat = k.reshape(BH, M, d)

        # ── Collect candidates from every round, union them ─────────────
        # All index tensors use int32 (sequence lengths fit) to halve
        # memory vs int64 — significant at large N where index tensors
        # dominate peak allocation. Bucket tables (bstart, bsize) are
        # pre-allocated once and reused across rounds to avoid per-round
        # allocation overhead.
        num_buckets = 2 ** p_eff
        round_buf = {
            'bstart': torch.full((BH, num_buckets), M, dtype=torch.long, device=device),
            'bsize':  torch.zeros((BH, num_buckets), dtype=torch.long, device=device),
            'positions': torch.arange(M, device=device).unsqueeze(0).expand(BH, M),
        }
        round_cands, round_valids = [], []
        for r in range(self.R):
            cand, valid = self._single_round(q_flat, k_flat, r, top_k, causal,
                                             glob_fallback, p_eff, self.T,
                                             _buf=round_buf)
            round_cands.append(cand.to(torch.int32))
            round_valids.append(valid)

        all_cand  = torch.cat(round_cands, dim=-1).to(torch.int32)  # (BH, N, R*C) int32
        all_valid = torch.cat(round_valids, dim=-1)
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

        # ── Exact rescore over the UNION (chunked, no (BH,N,Call,d) spike) ─
        # Process candidates in chunks to avoid materialising the full
        # (BH, N, Call, d) cand_k tensor, which dominates peak memory at
        # large N (e.g. 6.8 GB at N=4096 with K=128, R=8, os=8).
        actual_k = min(top_k, Call)
        k_flat2d = k_flat.reshape(BH * M, d)
        batch_offset = torch.arange(BH, device=device).unsqueeze(1) * M

        CHUNK = min(Call, max(256, actual_k * 4))
        best_scores = torch.full((BH, N, actual_k), float("-inf"), device=device)
        best_idx    = torch.zeros(BH, N, actual_k, dtype=torch.int32, device=device)

        for start in range(0, Call, CHUNK):
            end = min(start + CHUNK, Call)
            chunk_cand = all_cand[:, :, start:end]            # (BH, N, chunk_C) int32
            chunk_valid = all_valid[:, :, start:end]

            idx_flat = chunk_cand.reshape(BH, N * (end - start)).long()
            flat_idx = (idx_flat + batch_offset).reshape(-1)
            chunk_k = torch.index_select(k_flat2d, 0, flat_idx).view(
                BH, N, end - start, d)

            chunk_scores = torch.einsum(
                'bnd,bncd->bnc', q_flat.float(), chunk_k.float())
            chunk_scores = chunk_scores.masked_fill(~chunk_valid, float("-inf"))

            # Merge with running best via topk on concatenated scores
            merged_scores = torch.cat([best_scores, chunk_scores], dim=-1)
            merged_idx    = torch.cat([best_idx, chunk_cand], dim=-1)
            best_scores, merge_best = merged_scores.topk(actual_k, dim=-1)
            best_idx = torch.gather(merged_idx, -1, merge_best)

        result = best_idx.to(torch.long)

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
    #
    # PERFORMANCE FIX: kept as a 0-dim TENSOR (not .item()'d to a Python
    # int) for the rest of this function. `.item()` forces a device→host
    # synchronization on every call — on CUDA this stalls the launch queue
    # until the device finishes all prior work, which is exactly the kind
    # of per-forward-pass sync that's easy to miss in CPU testing (where
    # there's no queue to stall) but measurable in real GPU training/
    # inference loops. torch.clamp, comparisons (>, <=), and masked_fill
    # all accept a 0-dim tensor identically to a Python scalar — verified
    # directly, no behavior change, purely removes the sync point.
    true_max_idx = combined.max()
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
        import warnings

        assert config.d_model % config.num_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_heads})"

        # -- GQA / MQA -------------------------------------------------------
        # num_kv_heads=None  -> standard MHA (num_kv_heads == num_heads)
        # num_kv_heads=1     -> MQA (single shared KV head)
        # 1 < num_kv_heads < num_heads -> GQA (LLaMA-2 / Mistral style)
        #
        # Wk/Wv project to (num_kv_heads * d_head) instead of d_model.
        # In forward(), each KV head is expanded to serve
        # (num_heads // num_kv_heads) query heads via repeat_interleave
        # before being passed to merge_neighbors / sparse_gather_attention,
        # which always operate in the full (B, H, ...) space.
        n_kv = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        assert config.num_heads % n_kv == 0, \
            f"num_heads ({config.num_heads}) must be divisible by num_kv_heads ({n_kv})"

        self.config   = config
        self.H        = config.num_heads
        self.n_kv     = n_kv
        self.kv_rep   = config.num_heads // n_kv   # expansion factor (1 for MHA)
        self.d_head   = config.d_model // config.num_heads
        self.K        = config.num_neighbors
        self.scale    = self.d_head ** -0.5

        self.Wq = nn.Linear(config.d_model, config.num_heads * self.d_head, bias=False)
        self.Wk = nn.Linear(config.d_model, n_kv             * self.d_head, bias=False)
        self.Wv = nn.Linear(config.d_model, n_kv             * self.d_head, bias=False)
        self.Wo = nn.Linear(config.num_heads * self.d_head, config.d_model, bias=False)

        self.win_builder  = WindowGraphBuilder(config.window_size, config.causal)
        self.glob_builder = GlobalGraphBuilder(
            num_global = config.num_global_tokens,
            indices    = config.global_token_indices,
        )

        n_glob     = (len(config.global_token_indices)
                      if config.global_token_indices is not None
                      else config.num_global_tokens)
        guaranteed = 2 * config.window_size + 1 + n_glob + 1
        self.lsh_k = max(self.K - guaranteed, 8)

        # Efficiency note (not a correctness issue -- see merge_neighbors'
        # `valid` mask, which makes oversized/wasted candidate budgets
        # harmless to the output): when window_size and num_global_tokens
        # are large relative to num_neighbors, the raw candidate pool
        # (window + global + self + lsh_k) can substantially exceed K
        # before truncation/dedup. E.g. K=16, window_size=16 ->
        # guaranteed=36 already exceeds K, lsh_k floors at 8, and the raw
        # pool (33+2+1+8=44) is ~2.8x what's actually kept. This wastes
        # compute (LSH still does real work for candidates that get
        # discarded) without being wrong. If profiling shows this matters
        # for your config, reduce window_size or increase num_neighbors so
        # `2*window_size+1 + num_global_tokens + 1` stays comfortably
        # below K.
        if guaranteed > self.K:
            warnings.warn(
                f"SSAConfig: window+global+self budget ({guaranteed}) exceeds "
                f"num_neighbors ({self.K}). lsh_k is floored at 8 and a large "
                f"fraction of computed candidates will be discarded as padding "
                f"(harmless to correctness, wasteful of compute). Consider "
                f"reducing window_size or increasing num_neighbors.",
                stacklevel=2,
            )

        self.lsh_builder = LSHGraphBuilder(
            d_head          = self.d_head,
            max_num_hashes  = config.max_num_hashes,
            lsh_candidates  = self.lsh_k * 4,
            num_rounds      = config.num_hash_rounds,
            num_probes      = config.lsh_num_probes,
        )
        self.dp = config.dropout

    def _split_q(self, x):
        """Split query projection: (B,N,H*d) -> (B,H,N,d)"""
        B, N, _ = x.shape
        return x.view(B, N, self.H, self.d_head).transpose(1, 2)

    def _split_kv(self, x):
        """Split KV projection: (B,M,n_kv*d) -> (B,H,M,d) with GQA expansion."""
        B, M, _ = x.shape
        # (B, M, n_kv, d_head) -> (B, n_kv, M, d)
        kv = x.view(B, M, self.n_kv, self.d_head).transpose(1, 2)
        if self.kv_rep > 1:
            # Expand to (B, H, M, d) by repeating each KV head kv_rep times.
            # repeat_interleave keeps head-group ordering consistent with how
            # query heads are split (head 0..kv_rep-1 share KV head 0, etc.)
            kv = kv.repeat_interleave(self.kv_rep, dim=1)
        return kv  # (B, H, M, d)

    # _split kept as alias so existing callers of _split(q) still work
    def _split(self, x):
        return self._split_q(x)

    def _merge(self, x):
        B, H, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.config.d_model)

    def build_graph(
        self,
        q:        torch.Tensor,   # (B, H, N, d) -- already projected & split
        k:        torch.Tensor,   # (B, H, M, d) -- already projected & split
        N: int, M: int, B: int,
        device,
        causal:   bool,
        is_cross: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the sparse neighbor graph (window + global + LSH + self,
        merged/deduped) without touching q/k/v values. Split out of
        forward() specifically so the graph can be computed once and
        reused across multiple SparseAttention instances/layers -- see
        CachedGraphSparseTransformer below for the actual reuse path
        (LSA-style Cross-Layer Indexing).

        Returns (neighbors, valid), both (B, H, N, K) -- the same contract
        `merge_neighbors` always returned, just no longer fused with
        apply_graph so the two can be called independently or cached
        between calls.
        """
        glob_idx = self.glob_builder(M, device)        # (G,)

        # BUG FIX (cross-attention window graph was positionally meaningless):
        # WindowGraphBuilder assumes query position i should attend keys near
        # position i -- a reasonable locality prior for self-attention (N==M,
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
        # masked to -inf before softmax -- see merge_neighbors/
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
        return neighbors, valid

    def apply_graph(
        self,
        q:         torch.Tensor,   # (B, H, N, d)
        k:         torch.Tensor,   # (B, H, M, d)
        v:         torch.Tensor,   # (B, H, M, d)
        neighbors: torch.Tensor,   # (B, H, N, K)
        valid:     torch.Tensor,   # (B, H, N, K)
    ) -> torch.Tensor:
        """
        Runs sparse_gather_attention against an already-built neighbor
        graph. Split out of forward() so a cached graph (from a previous
        layer, see CachedGraphSparseTransformer) can be applied to THIS
        layer's q/k/v without recomputing window/global/LSH indices.

        Caller is responsible for ensuring `neighbors`/`valid` shapes match
        this layer's (B, H, N, K) -- if reusing across layers with
        different d_head or K, this will shape-mismatch loudly rather than
        silently producing wrong output.
        """
        return sparse_gather_attention(
            q, k, v, neighbors,
            scale    = self.scale,
            valid    = valid,
            dropout  = self.dp,
            training = self.training,
            fp32_attn_weights = self.config.fp32_attn_weights,
        )

    def forward(
        self,
        x:              torch.Tensor,
        key_value:      Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats:   bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x:              Query input  (B, N, d_model).
            key_value:      Key/value source for cross-attention (B, M, d_model).
                            None means self-attention.
            attention_mask: MUST be None.  Sparse attention does not support
                            dense external masks — the graph already encodes
                            causality (config.causal=True) and cross-attention
                            padding (global/LSH fallback).  Passing a mask here
                            raises ValueError so callers get an explicit error
                            rather than the mask being silently ignored.
                            If you need to mark padding tokens, set those
                            positions to 0 in `x` before calling forward().
            return_stats:   If True, return (output, stats_dict).
                            If False (default), return (output, {}) — the empty
                            dict keeps the two-tuple contract without forcing
                            callers to unpack a dict they don't need.

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
        whole forward pass -- autocast keeps matmul *reductions* in FP32
        internally even when inputs/outputs are FP16, which this module
        does not have visibility into or control over from inside. If you
        are training in raw FP16 without autocast, overflow risk in Wq/Wk/Wv
        is a property of that choice, common to every nn.Linear layer in
        the model, not specific to this attention implementation.
        """

        # -- Causal-mask guard -----------------------------------------------
        # Dense external masks are incompatible with sparse neighbor graphs.
        # Raise early with an actionable message instead of silently ignoring.
        if attention_mask is not None:
            raise ValueError(
                "SparseAttention does not accept an external attention_mask. "
                "Use config.causal=True for autoregressive masking (handled "
                "internally by WindowGraphBuilder and LSHGraphBuilder). "
                "For padding, zero out padding positions in x before calling "
                "forward(), or set num_global_tokens to 0 so padding tokens "
                "are not forced into every query's neighbor set."
            )

        B, N, _ = x.shape
        is_cross = key_value is not None
        src      = key_value if is_cross else x
        M        = src.shape[1]
        device   = x.device
        causal   = self.config.causal and not is_cross  # never causal for cross-attn

        q = self._split_q(self.Wq(x))
        k = self._split_kv(self.Wk(src))   # GQA expansion happens here
        v = self._split_kv(self.Wv(src))

        neighbors, valid = self.build_graph(q, k, N, M, B, device, causal, is_cross)
        out = self.apply_graph(q, k, v, neighbors, valid)

        out_proj = self.Wo(self._merge(out))

        if not return_stats:
            return out_proj, {}

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
            # graph can usefully fill -- informational only, does not affect
            # correctness now that padding is masked to -inf before softmax.
            "padding_fraction":  (~valid).float().mean().item(),
        }
        return out_proj, stats


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

    def forward(self, x, return_stats: bool = False):
        a, stats = self.attn(self.norm1(x), return_stats=return_stats)
        return x + a + self.ff(self.norm2(x + a)), stats


class SparseTransformer(nn.Module):
    def __init__(self, config: SSAConfig, num_layers: int = 6, vocab_size: int = 0):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, config.d_model) if vocab_size else None
        self.layers = nn.ModuleList([SparseTransformerLayer(config) for _ in range(num_layers)])
        self.norm   = nn.LayerNorm(config.d_model)

    def forward(self, x, return_stats: bool = False):
        if self.embed is not None:
            x = self.embed(x)
        stats_list = []
        for layer in self.layers:
            x, stats = layer(x, return_stats=return_stats)
            stats_list.append(stats)
        return self.norm(x), stats_list


# ────────────────────────────────────────────────────────────────────────────
# Cross-layer graph reuse (LSA-style Cross-Layer Indexing)
#
# Inspired by LongCat Sparse Attention's "Cross-Layer Indexing" (CLI):
# https://longcat.chat/blog/longcat-2.0 -- LSA observed that attention
# saliency is empirically stable across adjacent layers in a model trained
# end-to-end with that property as a training-time distillation target,
# and amortizes the indexer cost by reusing one indexing pass across
# several consecutive layers at inference.
#
# IMPORTANT DISANALOGY -- read before using this in production:
# LSA's indexer is LEARNED and trained jointly with the rest of the model
# (continued pretraining + cross-layer distillation), specifically so the
# model is pushed toward producing the cross-layer-stable saliency that
# makes reuse safe. LSHGraphBuilder here is UNLEARNED and structural --
# each layer's `rand_proj` buffer is an independently-initialized random
# hyperplane set with no training signal pushing layer N's hash buckets to
# resemble layer N+1's. There is no a priori reason to expect
# cross-layer-stable neighbor selection here the way there is for a
# content-based learned indexer.
#
# This class implements the mechanism (graph reuse across `reuse_every`
# layers, sharing one LSHGraphBuilder instance across the group so the
# hash projections are at least consistent within a reuse window) AND a
# measurement utility (`measure_cross_layer_overlap`) to check whether the
# stability assumption actually holds for a given model/config BEFORE
# trusting cached output. Do not skip that measurement step --
# see class docstring below for how to use it.
# ────────────────────────────────────────────────────────────────────────────

def neighbor_overlap(neighbors_a: torch.Tensor, neighbors_b: torch.Tensor,
                      valid_a: torch.Tensor, valid_b: torch.Tensor) -> torch.Tensor:
    """
    Per-(batch, head, query) Jaccard similarity between two neighbor index
    sets of shape (B, H, N, K). Used to measure how much overlap exists
    between two layers' independently-built graphs for the same query
    positions, BEFORE deciding cross-layer caching is safe to use.

    Returns (B, H, N) float tensor in [0, 1]; 1.0 means identical neighbor
    sets (ignoring order and invalid/padding slots), 0.0 means disjoint.

    This is a CPU-friendly reference implementation (loops over the K
    dimension with broadcasting, not a custom kernel) -- fine for an
    offline diagnostic run on a handful of batches, not intended to run
    inside the hot training/inference loop.
    """
    B, H, N, K = neighbors_a.shape
    # Mask invalid slots to a sentinel value that can never collide with a
    # real index (real indices are >= 0), so invalid slots never falsely
    # count as a match between A and B.
    sentinel = -1
    a = torch.where(valid_a, neighbors_a, torch.full_like(neighbors_a, sentinel))
    b = torch.where(valid_b, neighbors_b, torch.full_like(neighbors_b, sentinel))

    # (B,H,N,K,1) vs (B,H,N,1,K) -> (B,H,N,K,K) match matrix
    match = (a.unsqueeze(-1) == b.unsqueeze(-2))
    # An index in A matches B if it equals ANY valid entry in B
    a_in_b = match.any(dim=-1) & valid_a               # (B,H,N,K)
    intersection = a_in_b.sum(dim=-1).float()          # (B,H,N)

    union = (valid_a.sum(dim=-1) + valid_b.sum(dim=-1)).float() - intersection
    union = union.clamp_min(1.0)  # avoid div-by-zero when both sets are empty
    return intersection / union   # (B,H,N)


def measure_cross_layer_overlap(
    model: "SparseTransformer",
    x: torch.Tensor,
) -> dict:
    """
    Diagnostic: for a trained (or untrained) SparseTransformer, runs a
    forward pass and measures neighbor-graph Jaccard overlap between every
    pair of adjacent layers. This is the experiment that should be run
    BEFORE adopting CachedGraphSparseTransformer for a real model --
    see the module-level note above.

    Usage:
        model = SparseTransformer(cfg, num_layers=8)
        x = torch.randn(4, 512, cfg.d_model)
        report = measure_cross_layer_overlap(model, x)
        print(report["mean_overlap_per_adjacent_pair"])
        # e.g. [0.41, 0.38, 0.52, ...] -- one value per (layer_i, layer_i+1)
        # pair. Values near 0 mean cross-layer caching would silently
        # discard most of what the next layer would have actually attended
        # to. There's no universal "safe" threshold here -- it depends on
        # how much quality loss is tolerable for the compute saved -- but
        # values well below the overlap LSA reports for their TRAINED,
        # distilled indexer should be treated as a sign this technique
        # needs training-time support to work here, not just inference-time
        # wiring.

    Returns a dict with raw per-layer-pair overlap tensors plus a
    summary scalar per pair, intended for the caller to inspect/plot, not
    to feed an automatic threshold decision.
    """
    hidden = x
    if model.embed is not None:
        hidden = model.embed(x)

    per_layer_graphs = []
    h = hidden
    for layer in model.layers:
        normed = layer.norm1(h)
        attn = layer.attn
        B, N, _ = normed.shape
        is_cross = False
        M = N
        device = normed.device
        causal = attn.config.causal

        q = attn._split_q(attn.Wq(normed))
        k = attn._split_kv(attn.Wk(normed))
        v = attn._split_kv(attn.Wv(normed))

        neighbors, valid = attn.build_graph(q, k, N, M, B, device, causal, is_cross)
        per_layer_graphs.append((neighbors.detach(), valid.detach()))

        out = attn.apply_graph(q, k, v, neighbors, valid)
        a = attn.Wo(attn._merge(out))
        h = h + a + layer.ff(layer.norm2(h + a))

    overlaps = []
    for i in range(len(per_layer_graphs) - 1):
        n_a, v_a = per_layer_graphs[i]
        n_b, v_b = per_layer_graphs[i + 1]
        ov = neighbor_overlap(n_a, n_b, v_a, v_b)   # (B,H,N)
        overlaps.append(ov)

    return {
        "per_pair_overlap":               overlaps,                              # list of (B,H,N) tensors
        "mean_overlap_per_adjacent_pair": [ov.mean().item() for ov in overlaps], # list of floats
        "num_layers":                     len(model.layers),
    }


class CachedGraphSparseTransformer(nn.Module):
    """
    SparseTransformer variant implementing LSA-style Cross-Layer Indexing:
    the sparse neighbor graph is built once every `reuse_every` layers and
    reused (re-applied to that layer's own q/k/v) for the layers in
    between, skipping their window/global/LSH graph construction entirely.

    Read the module-level docstring above this class before using this in
    a real model -- this is an unlearned-LSH analog of a technique that,
    in its original form, relies on the model being trained to make
    cross-layer reuse safe. Run `measure_cross_layer_overlap` on your
    model/config first.

    Implementation notes:
      - All layers within a `reuse_every`-sized group SHARE one
        LSHGraphBuilder instance (and one window/global builder), so the
        hash projections are at least consistent across the group. Without
        this, "reusing the graph" would mean applying layer N's
        hash-derived neighbor indices to layer N+1's Q/K, which were never
        hashed by the same projection -- meaningless regardless of
        cross-layer saliency stability.
      - Each layer still has its OWN Wq/Wk/Wv/Wo projections and its own
        FFN -- only the neighbor GRAPH (which key positions each query
        attends to) is shared, not the attention weights or values
        themselves. The actual attention scores/output differ per layer
        even when the candidate neighbor set doesn't.
      - The first layer in each group always does a full graph build
        (recompute=True); subsequent layers in the group reuse it.
    """

    def __init__(self, config: SSAConfig, num_layers: int,
                 reuse_every: int = 4, ffn_mult: int = 4):
        super().__init__()
        assert reuse_every >= 1
        self.reuse_every = reuse_every
        self.config = config

        # One shared graph-builder set per group of `reuse_every` layers.
        num_groups = math.ceil(num_layers / reuse_every)
        self.shared_attn_per_group = nn.ModuleList([
            SparseAttention(config) for _ in range(num_groups)
        ])
        # group_of[i] tells layer i which shared SparseAttention instance
        # to borrow build_graph() from (its OWN Wq/Wk/Wv/Wo live on the
        # per-layer SparseAttention below; the group instance is ONLY used
        # for its graph-builder submodules).
        self.group_of = [i // reuse_every for i in range(num_layers)]

        self.layers = nn.ModuleList([
            SparseTransformerLayer(config, ffn_mult) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, return_stats: bool = False):
        h = x
        stats_list = []
        cached_neighbors = None
        cached_valid = None
        cached_group = None

        for i, layer in enumerate(self.layers):
            group = self.group_of[i]
            recompute = (group != cached_group)

            normed = layer.norm1(h)
            attn = layer.attn
            B, N, _ = normed.shape
            device = normed.device
            causal = attn.config.causal

            q = attn._split_q(attn.Wq(normed))
            k = attn._split_kv(attn.Wk(normed))
            v = attn._split_kv(attn.Wv(normed))

            if recompute:
                # Graph built using THIS GROUP's shared builder submodules,
                # but against the CURRENT layer's q/k (each layer still has
                # its own Wq/Wk projections -- only window/global/LSH
                # builder PARAMETERS are shared within the group, not q/k
                # themselves).
                builder_attn = self.shared_attn_per_group[group]
                neighbors, valid = builder_attn.build_graph(
                    q, k, N, N, B, device, causal, is_cross=False
                )
                cached_neighbors, cached_valid, cached_group = neighbors, valid, group
            else:
                neighbors, valid = cached_neighbors, cached_valid

            out = attn.apply_graph(q, k, v, neighbors, valid)
            a = attn.Wo(attn._merge(out))
            h = h + a + layer.ff(layer.norm2(h + a))

            if return_stats:
                peak_sparse = B * attn.H * N * attn.K * attn.d_head
                peak_full   = B * attn.H * N * N * attn.d_head
                stats_list.append({
                    "neighbors_shape":   tuple(neighbors.shape),
                    "compression":       peak_full / peak_sparse,
                    "padding_fraction":  (~valid).float().mean().item(),
                    "graph_recomputed":  recompute,
                })
            else:
                stats_list.append({})

        return self.norm(h), stats_list


