# SparseAttention

O(N·K) multi-head attention for PyTorch — a sparse drop-in replacement for
dense scaled-dot-product attention. Combines a local window, a fixed set of
global tokens, and content-based LSH routing into a single per-query
neighbor list, then attends only to that list instead of the full sequence.

```python
from sparse_attention import SSAConfig, SparseAttention

cfg = SSAConfig(d_model=512, num_heads=8, num_neighbors=128,
                window_size=8, num_global_tokens=2, causal=True)
attn = SparseAttention(cfg)

out, _ = attn(x)                     # self-attention, x: (B, N, d_model)
out, _ = attn(x, key_value=enc_out)  # cross-attention
```

## Why this exists

Dense attention is O(N²) in both compute and memory. At long context lengths
that dominates everything else in a transformer. The usual answer is either
a fixed sparsity pattern (local windows, strided attention — cheap but loses
long-range dependencies) or a *learned* content-based sparse indexer
(DeepSeek's DSA, LongCat's LSA — high quality but requires training the
indexer jointly with the model).

This module sits in between: it uses **unlearned, structural** routing
(random-hyperplane LSH, like the Reformer) so it works with any pretrained
model's weights with no indexer training step, while still being
content-aware rather than purely positional. The tradeoff is recall —
LSH-based candidate selection is approximate, not exact top-k, so there's a
real (measured, not assumed) gap versus what a trained indexer or exact
attention would select. See [Recall and quality](#recall-and-quality) below
for actual numbers, not just the claim.

## How the neighbor graph is built

For each query token, the candidate key set is the union of four sources,
deduplicated and padded/truncated to exactly `num_neighbors` slots:

1. **Self** — every token always attends to itself.
2. **Window** — the `2·window_size + 1` nearest positions (causal: only the
   trailing half).
3. **Global** — a fixed set of key positions every query attends to. Either
   the first `num_global_tokens` positions, or an explicit list via
   `global_token_indices` (e.g. `[0, 1]` for BOS/CLS, or landmark positions
   that aren't at the start of the sequence).
4. **LSH** — content-based candidates found via multi-round, multi-plane
   locality-sensitive hashing. Queries and keys are hashed into buckets by
   random hyperplane projections; candidates come from the query's own
   bucket, exactly rescored, and top-k'd.

The LSH stage is the part doing real algorithmic work — window and global
are simple index arithmetic with no false negatives. The LSH bucket count is
**adaptive**, not fixed: the number of hash planes used per forward call is
computed from the current sequence length so that average bucket occupancy
stays near the per-round candidate budget, which empirically maximizes
recall@K across sequence lengths from 64 to 32,768 tokens (see the
extensive in-code derivation in `LSHGraphBuilder` — this was tuned
adversarially against a fixed-plane-count baseline, not assumed correct).
Multiple independent hash rounds are unioned before the final rescore to
reduce the false-negative rate inherent to single-round LSH (measured ~9%
at 1 round, ~0% at 4 rounds in testing).

## Installation

```bash
pip install torch
```

No other dependencies. Single file, ~1900 lines including an extensive
inline test suite — copy `sparse_attention.py` into your project, or vendor
the whole repo.

## Configuration

All behavior is controlled by `SSAConfig`:

```python
@dataclass
class SSAConfig:
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None       # GQA/MQA — see below
    num_neighbors: int = 128                 # K: total neighbor slots per query
    max_num_hashes: int = 12                 # ceiling on LSH planes (2^P buckets)
    num_hash_rounds: int = 8                 # independent hash rounds, unioned
    lsh_num_probes: int = 0                  # multi-probe: extra near-boundary
                                              # buckets checked per round, see below
    window_size: int = 8                     # local window half-width
    num_global_tokens: int = 2               # leading key tokens, all queries attend
    global_token_indices: Optional[list] = None  # explicit global positions, overrides above
    dropout: float = 0.0
    causal: bool = False
    fp32_attn_weights: bool = False          # keep post-softmax weights in FP32
```

### Picking `num_neighbors` vs `window_size` + `num_global_tokens`

The module warns at construction time if `2*window_size + 1 + num_global_tokens + 1`
(the guaranteed, non-LSH candidate budget) exceeds `num_neighbors` — in that
regime LSH gets starved down to a floor of 8 candidates and a large fraction
of computed candidates get discarded as padding. Not incorrect, just
wasteful of compute. Keep the guaranteed budget comfortably under `K`.

A good rule of thumb for `window_size`: `max(1, K // 8)`. The window
provides positional locality that partially overlaps with LSH content-based
routing. Its value depends on the LSH coverage ratio (`lsh_k / N`): when
LSH candidates cover >10% of the sequence, the window is redundant. With
the evolved defaults (K=128, R=8), window size has <1% impact on recall
at all tested sequence lengths.

### GQA / MQA

```python
# Standard MHA (default)
SSAConfig(num_heads=8)                       # num_kv_heads defaults to num_heads

# Grouped-query attention (LLaMA-2 / Mistral style)
SSAConfig(num_heads=32, num_kv_heads=8)      # 4 query heads share each KV head

# Multi-query attention (single shared KV head, max memory savings)
SSAConfig(num_heads=32, num_kv_heads=1)
```

`num_heads` must be divisible by `num_kv_heads`. KV heads are expanded via
`repeat_interleave` before graph building, so all downstream sparse-routing
code is unaffected — GQA/MQA only changes the size of `Wk`/`Wv`.

### Non-leading global tokens

```python
SSAConfig(global_token_indices=[0, 1, 512])  # BOS + CLS + a mid-doc landmark
```

Overrides `num_global_tokens` entirely. Indices are clamped to `[0, M-1]` at
runtime, so a config built for one sequence length is still safe (if
wasteful) when applied to a shorter one.

## What's NOT supported

- **External attention masks.** Passing `attention_mask` raises
  `ValueError` with an explanation, rather than silently ignoring it (which
  is what happens if you pass a dense mask to a module that only
  understands its own sparse graph). Use `config.causal=True` for
  autoregressive masking; for padding, zero out padding positions in the
  input before calling forward, or set `num_global_tokens=0` so padded
  positions aren't forced into every query's neighbor set.
- **KV caching for incremental decoding.** This implementation rebuilds
  the full graph on every forward call. There is no append-one-token,
  reuse-cache path. Adding one would mean making `LSHGraphBuilder`
  incremental (new key gets hashed and inserted into existing buckets
  rather than re-bucketing everything), which is a real piece of unbuilt
  work, not a config flag.
- **Exact recall guarantees.** LSH is approximate. See below.

## Multi-probe LSH (`lsh_num_probes`)

By default a query only checks its own hash bucket per round; false
negatives near a bucket boundary are caught by `num_hash_rounds`
independent rehashing instead. `lsh_num_probes` adds a second, more
targeted way to catch the same failure mode: for each query, also check
the `lsh_num_probes` buckets one bit-flip away from its own, ranked by
which hash bit had the smallest-magnitude hyperplane projection (i.e.
probe the boundary this specific query is actually closest to, rather
than hoping an independent round's random projection happens to
separate a near-boundary pair correctly). Off by default
(`lsh_num_probes=0`); existing configs are unaffected.

```python
cfg = SSAConfig(..., num_hash_rounds=4, lsh_num_probes=1)
```

**Measured tradeoff** (`benchmarks/bench_multiprobe.py`, N=2048, K=64,
true_k=32, random Q/K — same synthetic-floor caveat as the recall
benchmark below): at a FIXED `num_hash_rounds=4`, probes are a genuine
recall lever — 48.5% (T=0) → 72.6% (T=1) → 91.3% (T=3) — but they are
not free: each probe adds another full candidate gather, the same
rescore cost as an extra round. At *matched* total rescore budget
(`R * (1 + T)`), spending it as more independent rounds slightly beats
spending it as probes on this synthetic random-embedding test (e.g.
R=4,T=0 at budget=4 hits 48.5% vs R=2,T=1's 47.1% — probing a specific
query's nearest boundary carries no extra signal when there's no real
structure to exploit, which random Gaussian Q/K by construction lacks).
Wall-clock is also close either way — probes reuse one round's bucket
sort instead of re-sorting for an independent round, which should in
principle save the ~32% of build cost this repo's profiling attributes
to `torch.sort`, but measured at N=4096 the saving was ~0.5%, since each
probe still pays its own gather/rescore cost in full and that's the
larger term. Net: use `lsh_num_probes` when you want to push recall
*beyond* what raising `num_hash_rounds` alone gives you and are willing
to pay for it, not as a drop-in efficiency swap for existing rounds. On
real trained embeddings (with actual cluster structure near bucket
boundaries, unlike this synthetic test) probing the query's own nearest
boundary may carry more signal than an independent reroll — not
measured here, flagged as an open question rather than assumed.

## Recall and quality

`tests/test_quality_and_scaling.py` includes a recall@K benchmark against
exact dense top-K attention on random embeddings:

```
N=1024, K=128, R=8, window=8, true_k=32
Sparse pipeline recall@32: 99.3%
Random-K-selection recall: 12.5%  (floor)
Ratio vs random floor: 7.94x
```

Recall across sequence lengths (K=128, R=8, window=8 vs old K=64, R=4,
window=16):

```
     N      Before      After       Gain
-------------------------------------------
    64      89.5%      99.2%      +9.7%
   128      85.9%      99.1%     +13.2%
   256      83.7%      99.6%     +16.0%
   512      73.3%      99.9%     +26.6%
  1024      54.7%      99.9%     +45.1%
  2048      38.1%      99.3%     +61.1%
  4096      27.0%      95.3%     +68.3%
```

99%+ recall against a *random, unstructured* embedding distribution
means the LSH routing selects neighbors that overlap almost perfectly
with exact dense top-K. On real trained representations (where similar
tokens cluster — exactly what LSH exploits), recall should be even
higher. This benchmark exists to catch *regressions* in the LSH routing
logic, not to predict production quality. If you change `LSHGraphBuilder`
and this ratio drops, something broke.

### AlphaEvolve tuning

The default configuration was optimized via an evolutionary search
(AlphaEvolve methodology) across the joint parameter space of
`num_neighbors`, `num_hash_rounds`, `window_size`, and LSH oversample
factor. Three rounds of evolution found:

| Parameter | Old default | Evolved default | Impact |
|-----------|-------------|-----------------|--------|
| `num_neighbors` | 64 | **128** | +29% recall at N=1024 |
| `num_hash_rounds` | 4 | **8** | +7% on top of K=128 |
| `window_size` | 16 | **8** | frees budget for LSH |

Combined effect: recall stays above 95% through N=4096, whereas the old
config collapsed below 50% around N=1500.

Key findings from the search:
- **K (num_neighbors) is the dominant lever.** The old default K=64 was
  undersized for long sequences. K=128 gives 92.6% recall at N=1024
  even with the old R=4.
- **R (hash rounds) is the second lever.** Each round adds an independent
  random hyperplane set; false-negative rate drops exponentially with R.
  R=8 pushes recall from 92.6% to 99.3%.
- **Smaller windows outperform larger ones.** window=8 beats window=16
  at all sequence lengths because a smaller window frees K budget for
  LSH content-based routing, which captures the important tokens more
  effectively than positional proximity.
- **Hash projection distribution doesn't matter.** Sign-based hashing
  `(proj > 0)` is inherently scale-invariant — scaling, normalizing, or
  power-transforming the projections produces identical bucket assignments.
- **Union strategy and rescore mode don't matter.** First-occurrence
  cross-round dedup and dot-product rescore are already optimal.

See the `alpha_evolve*.py` scripts (run during development, not shipped)
for the full evolutionary search code.

Memory scaling, same test suite, exact O(N²) vs O(NK) elements:

```
        N      Full elems    Sparse elems     Ratio
      256       8,388,608       1,048,576        8×
    1,024     134,217,728       4,194,304       32×
    4,096   2,147,483,648      16,777,216      128×
   16,384  34,359,738,368      67,108,864      512×
```

Compression scales with N as expected for a fixed K — the longer the
sequence, the larger the win.

## Benchmarks

`benchmarks/bench_dense_vs_sparse.py` compares `SparseAttention` against
`F.scaled_dot_product_attention` directly — wall-clock and the same
tensor-size scaling shown above, side by side.

```bash
python benchmarks/bench_dense_vs_sparse.py
```

**Headline result, measured on CPU (single core, no CUDA), not assumed:**

```
     N |   Dense (ms) |  Sparse (ms) |  Speedup | Dense tensor (MB) | Sparse tensor (MB) | Mem ratio
   256 |         3.94 |        62.39 |    0.06x |              1.05 |               0.52 |       2.0x
   512 |         7.08 |       126.09 |    0.06x |              4.19 |               1.05 |       4.0x
  1024 |        17.44 |       259.18 |    0.07x |             16.78 |               2.10 |       8.0x
  2048 |        56.02 |       544.34 |    0.10x |             67.11 |               4.19 |      16.0x
  4096 |       202.81 |      1131.41 |    0.18x |            268.44 |               8.39 |      32.0x
  8192 |       782.37 |      2467.90 |    0.32x |           1073.74 |              16.78 |      64.0x
```

**Dense is currently faster on this hardware at every N tested**, though
the gap narrows sharply as N grows (15x slower at N=256 → ~3x slower at
N=8192 — extrapolate the trend). This is the opposite of the usual sparse-
attention pitch, and it's reported here rather than worked around, because
it's true and explains something real.

The `run_profile_breakdown` function in the same script shows why: roughly
**55% of `SparseAttention`'s CPU time goes to `aten::index_select`** — the
gather operations that pull selected neighbor candidates out of K/V.
Gather has poor cache locality and doesn't vectorize the way a dense
matmul does; `F.scaled_dot_product_attention` on CPU is backed by a fused,
heavily optimized kernel that dense attention gets essentially for free by
being one big matmul. `SparseAttention` pays a real per-element indexing
tax that dense attention doesn't, because this repo uses plain `torch`
indexing rather than a custom gather/scatter kernel.

This is a property of *this implementation* (portable, dependency-free,
pure PyTorch) on *this hardware* (CPU), not evidence that sparse attention
as an idea is slower than dense. The actual payoff — avoiding O(N²) memory
and compute — only becomes a net wall-clock win once N is large enough
that dense attention's quadratic cost outweighs the gather tax, and/or on
hardware with a fast gather/scatter path (GPU + a custom Triton/CUDA
kernel, which is what production systems like FlashAttention-derived
sparse variants and DeepSeek's DSA actually ship). If you need a wall-clock
win at moderate N on CPU, this implementation as written won't give you
one — that's the honest result of actually running it, not a claim made
in either direction without measurement.

## Architecture / integration

`SparseAttention` is a self-contained module: it owns its own `Wq/Wk/Wv/Wo`
projections rather than taking pre-projected tensors, so it isn't a literal
kernel-level swap for `F.scaled_dot_product_attention`. Integrating into an
existing model (HuggingFace Transformers, a custom PyTorch model,
encoder-decoder cross-attention) means writing a thin adapter that copies
existing weights into `attn.Wq/Wk/Wv/Wo` and forwards the call —
straightforward but not zero-code. `SparseTransformerLayer` and
`SparseTransformer` are provided as ready-to-use building blocks if you're
not retrofitting an existing model.

```python
from sparse_attention import SparseTransformer

model = SparseTransformer(cfg, num_layers=6, vocab_size=32000)
out, stats_per_layer = model(token_ids)                     # stats=[{}]*6
out, stats_per_layer = model(token_ids, return_stats=True)  # populated stats
```

`return_stats=True` returns, per layer: `neighbors_shape`,
`peak_sparse_elems`, `peak_full_elems`, `compression` (ratio of the two),
and `padding_fraction` (fraction of neighbor slots that were unused
padding — high values mean `num_neighbors` is set larger than the graph
can usefully fill).

## Cross-layer graph caching (experimental)

`CachedGraphSparseTransformer` implements a technique inspired by
[LongCat-2.0's Cross-Layer Indexing](https://longcat.chat/blog/longcat-2.0):
share one neighbor graph across several consecutive layers instead of
rebuilding it at every layer, amortizing the LSH construction cost.

```python
from sparse_attention import CachedGraphSparseTransformer, measure_cross_layer_overlap

model = CachedGraphSparseTransformer(cfg, num_layers=12, reuse_every=4)
out, stats = model(x, return_stats=True)
# stats[i]["graph_recomputed"] tells you which layers actually rebuilt the graph
```

**Read this before using it.** LongCat's LSA indexer is *learned*, trained
jointly with the model so that the model is pushed toward producing
cross-layer-stable attention patterns — that's *why* reuse is safe there.
`LSHGraphBuilder` is unlearned and structural: each layer's hash planes are
independently initialized with no training signal encouraging similarity
between layers. There is no a priori reason this technique transfers.

Before trusting `CachedGraphSparseTransformer` on a real model, run:

```python
report = measure_cross_layer_overlap(model, x)
print(report["mean_overlap_per_adjacent_pair"])
```

This measures the actual Jaccard overlap between adjacent layers' neighbor
sets. On an untrained model this measured **~0.47** (close to what random
LSH bucket assignment would produce) — i.e. the stability assumption did
*not* hold by default, which is the expected and honest result, not a bug.
Don't skip this measurement and assume the technique works; it's wired up
specifically so you don't have to assume.

## Development history

This repo's git history is structured as one commit per design iteration —
13 commits walking from an initial O(N²)-adjacent draft through to the
current adaptive multi-round LSH implementation, followed by a fixes commit
(GQA/MQA, explicit global token indices, the attention-mask guard,
opt-in stats) and the cross-layer caching feature. `git log -p` is a
reasonably readable design narrative if you want to see *why* specific
choices were made and what they replaced — most non-trivial design
decisions in the code are accompanied by an inline comment explaining the
alternative that was tried and rejected, with numbers where available,
rather than presenting the current approach as the only one considered.

## Testing

```bash
pip install pytest
pytest
```

Tests live in `tests/`, organized by what they cover rather than mirroring
file structure:

| File | Covers |
|------|--------|
| `test_attention_shapes.py` | Self/cross-attention shape correctness, `return_stats` contract |
| `test_neighbor_graph.py` | Deduplication, self-edge placement, empty-bucket fallback |
| `test_causal.py` | No future-token leakage in LSH or window candidates |
| `test_gradients.py` | Gradient flow through the sparse gather/scatter machinery |
| `test_quality_and_scaling.py` | O(NK) vs O(N²) memory scaling, recall@K vs exact dense top-K |
| `test_gqa_mqa.py` | Grouped/multi-query attention parameter counts and correctness |
| `test_global_tokens_and_mask_guard.py` | Explicit `global_token_indices`, the `attention_mask` rejection guard |
| `test_multiprobe_lsh.py` | `lsh_num_probes` correctness (distinct probe ids, causal masking, gradient flow) and the recall-increases-with-probes regression floor |
| `test_build_apply_graph_split.py` | `build_graph()`/`apply_graph()` produce bit-identical output to `forward()` |
| `test_cached_graph_transformer.py` | `CachedGraphSparseTransformer`, the `neighbor_overlap` diagnostic |

The `test_quality_and_scaling.py::test_recall_at_k_beats_random_selection`
test is the single most important one in the suite: every other test
proves the routing machinery is mechanically correct (no crashes, no NaN,
no out-of-range or duplicate indices) but says nothing about whether the
selected neighbors are any *good*. This is the test that closes that gap
— see [Recall and quality](#recall-and-quality) above for what the numbers
mean and don't mean.

`test_cached_graph_transformer.py::test_measure_cross_layer_overlap_on_untrained_model_is_not_near_one`
is a deliberate guard against the cross-layer caching feature silently
becoming unsafe-by-default — see
[Cross-layer graph caching](#cross-layer-graph-caching-experimental) above.

## License

No license file included — treat as all-rights-reserved until the author
adds one.
