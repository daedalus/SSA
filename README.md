# SparseAttention

O(N·K) multi-head attention for PyTorch — a sparse drop-in replacement for
dense scaled-dot-product attention. Combines a local window, a fixed set of
global tokens, and content-based LSH routing into a single per-query
neighbor list, then attends only to that list instead of the full sequence.

```python
from sparse_attention import SSAConfig, SparseAttention

cfg = SSAConfig(d_model=512, num_heads=8, num_neighbors=64,
                window_size=16, num_global_tokens=2, causal=True)
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
    num_neighbors: int = 64                  # K: total neighbor slots per query
    max_num_hashes: int = 12                 # ceiling on LSH planes (2^P buckets)
    num_hash_rounds: int = 4                 # independent hash rounds, unioned
    window_size: int = 16                    # local window half-width
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

## Recall and quality

`tests/test_quality_and_scaling.py` includes a recall@K benchmark against
exact dense top-K attention on random embeddings:

```
N=1024, K=64, true_k=32
Sparse pipeline recall@32: 65.4%
Random-K-selection recall: 6.1%  (floor)
Ratio vs random floor: 10.68x
```

Read this correctly: 65% recall against a *random, unstructured* embedding
distribution is not a number to expect on real trained representations
(real attention has structure — similar tokens cluster, which is exactly
what LSH exploits — so real-world recall should be substantially higher
than this synthetic floor test). This benchmark exists to catch
*regressions* in the LSH routing logic, not to predict production quality.
If you change `LSHGraphBuilder` and this ratio drops, something broke.

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
