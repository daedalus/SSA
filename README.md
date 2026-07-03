# SparseAttention

O(N·K) multi-head attention for PyTorch — a sparse drop-in replacement for
dense scaled-dot-product attention. Combines a local window, a fixed set of
global tokens, and content-based LSH routing into a single per-query
neighbor list, then attends only to that list instead of the full sequence.

## Quick Start

### Installation

No extra dependencies — just PyTorch:

```bash
pip install torch
```

Then copy `sparse_attention.py` into your project, or clone the repo:

```bash
git clone https://github.com/daedalus/SSA.git
cd SSA
```

### Basic Usage

```python
import torch
from sparse_attention import SSAConfig, SparseAttention

# Configure and create the attention module
cfg = SSAConfig(d_model=512, num_heads=8, num_neighbors=128,
                window_size=8, num_global_tokens=2, causal=True)
attn = SparseAttention(cfg)

# Self-attention
x = torch.randn(2, 1024, 512)  # (batch, seq_len, d_model)
out, _ = attn(x)

# Cross-attention (e.g., encoder-decoder)
enc_out = torch.randn(2, 512, 512)  # encoder output
out, _ = attn(x, key_value=enc_out)
```

### Grouped-Query Attention (GQA / MQA)

```python
# Standard MHA (default)
cfg = SSAConfig(num_heads=8)

# Grouped-query attention (LLaMA-2 / Mistral style)
cfg = SSAConfig(num_heads=32, num_kv_heads=8)  # 4 query heads share each KV head

# Multi-query attention (single shared KV head, max memory savings)
cfg = SSAConfig(num_heads=32, num_kv_heads=1)
```

### Full Transformer

```python
from sparse_attention import SparseTransformer

model = SparseTransformer(cfg, num_layers=6, vocab_size=32000)

token_ids = torch.randint(0, 32000, (2, 1024))
out, stats_per_layer = model(token_ids)                     # no stats
out, stats_per_layer = model(token_ids, return_stats=True)  # with stats
```

### Explicit Global Token Indices

```python
# Force attention to BOS, CLS, and a mid-document landmark
cfg = SSAConfig(global_token_indices=[0, 1, 512])
```

## Configuration

All behavior is controlled by `SSAConfig`:

```python
@dataclass
class SSAConfig:
    d_model: int = 512
    num_heads: int = 8
    num_kv_heads: Optional[int] = None       # GQA/MQA
    num_neighbors: int = 128                 # K: total neighbor slots per query
    max_num_hashes: int = 12                 # ceiling on LSH planes (2^P buckets)
    num_hash_rounds: int = 8                 # independent hash rounds, unioned
    lsh_num_probes: int = 0                  # multi-probe: extra near-boundary buckets
    window_size: int = 8                     # local window half-width
    num_global_tokens: int = 2               # leading key tokens, all queries attend
    global_token_indices: Optional[list] = None  # explicit global positions
    dropout: float = 0.0
    causal: bool = False
    fp32_attn_weights: bool = False          # keep post-softmax weights in FP32
```

### Picking `num_neighbors` vs `window_size` + `num_global_tokens`

Keep the guaranteed budget (`2*window_size + 1 + num_global_tokens + 1`)
comfortably under `num_neighbors`. A good rule of thumb for `window_size`:
`max(1, K // 8)`.

## How It Works

For each query token, the candidate key set is the union of four sources:

1. **Self** — every token always attends to itself
2. **Window** — the `2·window_size + 1` nearest positions (causal: only the trailing half)
3. **Global** — a fixed set of key positions every query attends to
4. **LSH** — content-based candidates found via multi-round, multi-plane locality-sensitive hashing

The LSH bucket count is **adaptive** — computed from sequence length to keep average bucket occupancy near the per-round candidate budget, maximizing recall across sequence lengths from 64 to 32,768 tokens.

## Recall and Quality

Benchmarked against exact dense top-K attention on random embeddings:

```
N=1024, K=128, R=8, window=8, true_k=32
Sparse pipeline recall@32: 99.3%
Random-K-selection recall: 12.5%  (floor)
Ratio vs random floor: 7.94x
```

Recall across sequence lengths:

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

## Memory Efficiency

Peak LSH rescore memory (K=128, R=8, os=8):

```
     N      Before      After      Saved
-------------------------------------------
   512      0.98 GB     0.14 GB    7×
  1024      1.97 GB     0.28 GB    7×
  2048      3.93 GB     0.56 GB    7×
  4096      7.87 GB     1.12 GB    7×
```

## Benchmarks

```bash
python benchmarks/bench_dense_vs_sparse.py
```

**CPU performance (single core):**

```
     N |   Dense (ms) |  Sparse (ms) |  Speedup | Mem ratio
   256 |         3.94 |        62.39 |    0.06x |       2.0x
   512 |         7.08 |       126.09 |    0.06x |       4.0x
  1024 |        17.44 |       259.18 |    0.07x |       8.0x
  2048 |        56.02 |       544.34 |    0.10x |      16.0x
  4096 |       202.81 |      1131.41 |    0.18x |      32.0x
  8192 |       782.37 |      2467.90 |    0.32x |      64.0x
```

Dense is currently faster on CPU at every N tested, though the gap narrows sharply as N grows. The memory savings (up to 64×) become the dominant advantage for long sequences. On GPU with custom kernels, sparse attention's O(N·K) complexity provides both speed and memory wins.

## What's NOT Supported

- **External attention masks** — use `config.causal=True` for autoregressive masking; for padding, zero out positions before calling forward
- **KV caching for incremental decoding** — the full graph is rebuilt on every forward call
- **Exact recall guarantees** — LSH is approximate, not exact top-k

## Testing

```bash
pip install pytest
pytest
```

| Test File | Covers |
|-----------|--------|
| `test_attention_shapes.py` | Self/cross-attention shape correctness |
| `test_neighbor_graph.py` | Deduplication, self-edge placement |
| `test_causal.py` | No future-token leakage |
| `test_gradients.py` | Gradient flow through sparse operations |
| `test_quality_and_scaling.py` | Memory scaling, recall@K benchmarks |
| `test_gqa_mqa.py` | Grouped/multi-query attention |
| `test_global_tokens_and_mask_guard.py` | Explicit global indices, mask rejection |
| `test_multiprobe_lsh.py` | Multi-probe LSH correctness |
| `test_build_apply_graph_split.py` | build_graph/apply_graph consistency |
| `test_cached_graph_transformer.py` | Cross-layer graph caching |

## License

MIT — see [LICENSE](LICENSE) for details.
