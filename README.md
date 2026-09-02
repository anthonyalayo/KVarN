[![Built on vLLM](https://img.shields.io/badge/Built%20on-vLLM%20v0.28.0-30a14e)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2606.03458-b31b1b.svg)](https://arxiv.org/abs/2606.03458)
[![hf-space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Huawei%20CSL-ffc107?color=ffc107&logoColor=white)](https://huggingface.co/huawei-csl)
[![GitHub stars](https://img.shields.io/github/stars/huawei-csl/KVarN?label=Stars&logo=github&logoColor=white&style=flat-square)](https://github.com/huawei-csl/KVarN/stargazers)

This is KVarN rebased onto vLLM v0.28.0 (upstream was v0.23.0).

**My Quick Start:**

```bash
# 1. Clone
git clone git@github.com/anthonyalayo/KVarN.git
cd KVarN

# 2. Setup uv python environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12

# 3. Install (tweak the max jobs number to handle OOMs)
MAX_JOBS=16 uv pip install -e . -v

# 4. Start
source .venv/bin/activate
vllm serve <args>
```

**My Quick Serve (for RTX 5090 32GB VRAM):**

```bash
# WINNER: MTP+2 — 310.9 out tok/s (1.28× no-MTP), 1.82× 262K concurrency
# GPU KV cache size: 476,878 tokens, Maximum concurrency for 262,144 tokens per request: 1.82x
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 \
    --quantization modelopt --chat-template ~/qwen38-froggeric-v22.jinja \
    --kv-cache-dtype kvarn_k4v2_g128 --max-model-len auto --max-num-seqs 4 \
    --kv-cache-memory 9551856271 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --limit-mm-per-prompt '{"image": 4}' --mm-processor-kwargs '{"max_pixels": 8388608}'

# baseline: no MTP — 243.5 out tok/s, 2.07× 262K concurrency
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 \
    --quantization modelopt --chat-template ~/qwen38-froggeric-v22.jinja \
    --kv-cache-dtype kvarn_k4v2_g128 --max-model-len auto --max-num-seqs 4 \
    --kv-cache-memory 9551856271 \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --limit-mm-per-prompt '{"image": 4}' --mm-processor-kwargs '{"max_pixels": 8388608}'

# DFlash2 — 160.6 out tok/s, 0.6% acceptance; kept for reference, do not use
vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 \
    --quantization modelopt --chat-template ~/qwen38-froggeric-v22.jinja \
    --kv-cache-dtype kvarn_k4v2_g128 --max-model-len auto --max-num-seqs 4 \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"method":"dflash","model":"syvai/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --limit-mm-per-prompt '{"image": 4}' --mm-processor-kwargs '{"max_pixels": 8388608}'
```

**Note: `--kv-cache-memory 9551856271` gets a gpu util between 0.95 and 0.96 on the MTP+2 case.**

**Model note:** use the `-LMHead4` repo. The plain `-RTX5090` repo was
re-pushed and its current snapshot ships no MTP head tensors (and
`--revision` did not pin the old snapshot); the server boots fine on it,
but MTP acceptance is 0.0%.

What's different from upstream KVarN:

- Rebased onto vLLM 0.28.0 (upstream KVarN targets 0.23.0). The kvarn_ KV
  dtypes are rewired into 0.28's KV-cache plumbing: each dtype gets its own
  KVQuantMode, and the backend reports the packed slot size through 0.28's
  customize_spec hook (0.28 dropped the packed-spec classes the 0.27 build
  had been reusing).
- Hybrid (GDN/mamba) models now work with MTP: block and page sizes come from
  KVarN's packed layout instead of the plain-attention formula (which sized
  the mamba page wrong and crashed at boot), and the attention metadata is
  tracked in KVarN tiles (25x128 tokens per hybrid block) instead of raw
  block ids (which read out of bounds and produced garbage KV).
- MTP spec-decode verify: each KV block is dequantized once for all draft
  tokens (the shared-dequant kernel) instead of once per token, which cuts
  verify KV reads ~3x at long context. ON by default; set
  KVARN_SHARED_VERIFY=0 to revert (perf table below).
- DEV_NOTES.md: how to reinstall the flashinfer cubin after a base bump (the
  cached one goes stale).

dense and MLA attention are otherwise unchanged from upstream.

**Speculative decode profiling with "vllm bench serve --port 8000"**

All experiments with the above quick-serve commands on the LMHead4 model
(`--kv-cache-memory 9551856271` ≈ 0.95 gpu util), 1000 requests at
1024 input / 128 output tokens each.

| | No MTP | MTP+1 | MTP+2 | MTP+3 | DFlash2 (7) |
| --- | --- | --- | --- | --- | --- |
| Output tok/s | 243.5 | 296.0 | 310.9 | **316.3** | 160.6 |
| Total tok/s | 2191.5 | 2663.7 | 2797.9 | **2846.5** | 1445.4 |
| vs no MTP | 1.00× | 1.22× | 1.28× | **1.30×** | 0.66× |
| Mean TPOT (ms) | 15.52 | 12.60 | 11.88 | **11.58** | 23.35 |
| Median ITL (ms) | **14.22** | 16.86 | 18.97 | 20.05 | 21.47 |
| P99 ITL (ms) | **72.91** | 89.45 | 93.52 | 95.55 | 95.0 |
| KV cache (tokens) | 542,161 | 492,600 | 476,878 | 462,130 | — |
| 262K concurrency | 2.07× | 1.88× | 1.82× | 1.76× | — |
| Acceptance (len) | — | 59.7% (1.60) | 46.6% (1.93) | 37.2% (2.12) | 0.6% (1.04) |

**Winner: MTP+2.** It gets 98% of MTP+3's speed (310.9 vs 316.3 tok/s)
while keeping more KV cache free (1.82x vs 1.76x concurrency at 262K) and
lower inter-token latency (18.97 vs 20.05 median ITL). MTP+3 is only
faster on raw throughput, and it does so with the smallest KV cache and
the worst latency. DFlash2 does not work here: its W4A16 draft model gets
accepted only 0.6% of the time, so it ends up 34% slower than running
with no speculative decoding.

**Shared-dequant MTP verify (on by default):** when MTP checks its draft
tokens it normally re-reads the KV cache once per draft token. The
shared-dequant kernel reads each KV block once and reuses it for all the draft
tokens, so the verify step does about a third of the KV reads. Measured on a
single-process microbench (one request, no speculative-decoding acceptance
noise):

| ctx | shared (ms) | per-token (ms) | Δ |
| --- | --- | --- | --- |
| 16K | 0.069 | 0.131 | **−47%** |
| 32K | 0.128 | 0.245 | **−48%** |
| 64K | 0.241 | 0.378 | **−36%** |
| 128K | 0.536 | 0.817 | **−34%** |

The same comparison end-to-end in serving (MTP+2, `vllm bench serve`, random
prompts, 128 output tokens, c1, same seed for both boots):

| ctx | TPOT shared (ms) | TPOT per-token (ms) | Δ (ms/token) |
| --- | --- | --- | --- |
| 16K | 10.46 | 11.02 | **−0.56** |
| 32K | 11.26 | 12.63 | **−1.37** |
| 64K | 11.56 | 14.11 | **−2.56** |
| 131K | 14.01 | 16.99 | **−2.98** |

---

<p align="center">
  <img src="imgs/logo_600.png" alt="KVarN" width="640">
</p>

> ⚡️ **Built for agentic and long-context workloads.**

> 💡 KVarN delivers **3-5x more KV-cache capacity** and **up to ~1.3x the throughput** of FP16, so you fit far longer contexts and serve more concurrent requests, with **FP16-level accuracy**.

> 🔌 **Calibration-free, plug-and-play with vLLM.** A native vLLM attention backend: add one flag, no model changes, no calibration.

> 🥊 **Up to ~2.4× TurboQuant throughput**, same capacity, **higher accuracy**.

---

## Why KVarN (Variance Normalized KV-Cache)?

> **kvarn** /kvɑːɳ/ &nbsp;·&nbsp; *noun* (Swedish)
>
> 1. A grinding apparatus used to reduce substances into smaller particles or
>    powder, especially grains, seeds, spices, coffee beans, KV-caches.

KV-cache quantization usually comes with a catch. As the
[vLLM TurboQuant blog](https://vllm.ai/blog/2026-05-11-turboquant) shows, existing
methods buy extra KV-cache capacity but **give up throughput** (TurboQuant reports
**40 to 52% lower throughput** for 2.3-3.7x capacity), and aggressive low-bit
quantization also tends to **cost accuracy**. Losing both speed *and* quality is
the main reason KV-cache quantization is rarely turned on in production.

**KVarN is built to keep both.** On Qwen3-32B (AIME25, 16K-context burst, TP=2) it
matches FP16 accuracy and **beats its throughput** while delivering ~4× the KV-cache capacity:

<p align="center">
  <img src="imgs/pareto_qwen3-32b.png" alt="KVarN vs FP16 vs TurboQuant: accuracy, throughput and capacity" width="660">
</p>

KVarN stays in the upper-right corner the blog's methods can't reach: **FP16-level
accuracy, FP16-or-better throughput, and several times the context.**

---

## Quickstart

KVarN ships as a vLLM fork. Install it like vLLM, then select the KVarN KV-cache dtype.

```bash
# 1. Clone
git clone https://github.com/huawei-csl/KVarN.git
cd KVarN

# 2. Install (uses the upstream precompiled wheel; KVarN kernels are Triton, JIT-compiled at runtime)
VLLM_USE_PRECOMPILED=1 pip install -e .
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    dtype="float16",                    # KVarN runs in float16
    kv_cache_dtype="kvarn_k4v2_g128",   # enable KVarN
    block_size=128,                     # KVarN tile size
)
print(llm.generate("Explain KV-cache quantization in one sentence.",
                    SamplingParams(max_tokens=64))[0].outputs[0].text)
```

Serving works the same way:

```bash
vllm serve Qwen/Qwen3-32B --dtype float16 --kv-cache-dtype kvarn_k4v2_g128 --block-size 128
```

> **Note:** KVarN runs in `float16` compute. One vLLM block is one KVarN tile, so
> the tile / page size equals `--block-size`. Both **128** (default) and **64** are
> supported, selected by the matching preset (`kvarn_k4v2_g128` / `kvarn_k4v2_g64`).
> 128 is the design point; 64 gives finer quantization granularity at the cost of a
> little KV capacity (more per-tile scale overhead per token), at essentially the
> same throughput.

> **Tip (capacity):** KVarN realizes its full KV-cache capacity when there is room
> to amortize a small fixed decode workspace. On multi-GPU or generous
> `--gpu-memory-utilization` setups this is automatic. On a tight single-GPU budget,
> vLLM's CUDA-graph memory profiler can over-reserve and shrink the KV pool; set
> `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` (and/or raise
> `--gpu-memory-utilization`) to recover the full capacity.

### MLA models

KVarN also supports **Multi-head Latent Attention (MLA)**. To the best of our knowledge, **KVarN is the first vLLM-compatible sub-8-bit KV-cache quantization method to support MLA-based models**: it quantizes the compressed KV latent to int4. Use the **same** `--kv-cache-dtype` you use on
dense models; on an MLA model it automatically routes to the MLA latent path (no
code or env changes), and the fast path is on by default:

```bash
vllm serve zai-org/GLM-4.7-Flash \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --tensor-parallel-size 2
```

**GLM-4.7-Flash, KVarN vs bf16** (TP=2):

| Metric | bf16 | KVarN | KVarN / bf16 |
| --- | --- | --- | --- |
| Burst throughput @32K (tok/s) | 401 | 377 | **0.94×** |
| KV-cache capacity (tokens) | 313K | 865K | **2.77×** |
| AIME25 accuracy | 53.3% | **53.3%** | parity |

The win is **2.77× KV capacity at ~parity accuracy**. MLA's
latent is already tiny so KVarN is not a latency play there, but it lets you fit
far more concurrent context in the same memory. 

### Hybrid models (Mamba / linear-attention)

KVarN supports **hybrid models** that interleave linear-attention or Mamba layers
with standard full-attention layers, such as Qwen3.6-27B. KVarN compresses only the
full-attention layers (the layers that hold a KV cache); the linear-attention and
Mamba layers keep their own recurrent state and are left untouched. The fp16 decode
pool is sized from the full-attention layer count only, so a hybrid model loads with
the default flags, no manual pool tuning.

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128
```

The KV-cache capacity gain applies to the full-attention layers (where the cache
lives), so a hybrid model fits far more concurrent context in the same memory at near
parity accuracy.

### Speculative decoding (MTP, DFlash, and draft models)

KVarN is compatible with **speculative decoding**, including Multi-Token Prediction
(MTP). Pass `--speculative-config` exactly as you normally would, alongside the KVarN
`--kv-cache-dtype`:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The speculative verify step attends over the full cached context (KVarN reconstructs
it from the quantized cache), and a block is committed to the quantized cache only
once all of its tokens are accepted, so rejected draft tokens never corrupt history.

> **Note (weight quantization):** KVarN quantizes the KV cache independently of the
> model weights, so it composes with weight-quantized checkpoints (for example
> `compressed-tensors` / AWQ INT4) and MTP at the same time. Validated on Qwen3.6-27B
> in both `bfloat16` and AWQ INT4.

**DFlash.** KVarN also supports [DFlash](https://github.com/z-lab/dflash), a
parallel-drafting method whose drafter attends to the cached context with *non-causal*
(bidirectional) cross-attention. KVarN's backend advertises non-causal support, so the
drafter reads the quantized cache exactly as the target model does, with no extra flags
beyond the usual `--speculative-config`:

```bash
vllm serve Qwen/Qwen3.6-27B \
    --dtype bfloat16 \
    --kv-cache-dtype kvarn_k4v2_g128 \
    --block-size 128 \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'
```

Each drafting step issues a block of query tokens (the bonus token plus the speculative
tokens) that attend over the *entire* cached context rather than a causal prefix; KVarN
serves that block against the quantized cache and reconstructs the context the same way
it does for ordinary decode. As with MTP, a block is committed to the quantized cache
only once its tokens are accepted, so rejected drafts never corrupt history.

---

## How does KVarN work?

<p align="center">
  <img src="imgs/kvarn_pipeline.gif" alt="KVarN pipeline: Cache, Rotated Cache, Normalized Cache, Quantized Cache" width="760">
</p>

KVarN quantizes the KV cache one fixed-size token tile at a time, walking each tile
through the four stages above:

1. **Cache**: the raw fp16 KV tile (channels × tokens), straight from attention.

2. **Rotated Cache**: a **Hadamard rotation** along the channel dimension mixes
   channels so that per-channel outliers are spread out, making the tile easier to
   quantize. The rotation is orthonormal, so attention scores are preserved.

3. **Normalized Cache**: **iterative variance normalization** (Sinkhorn-like)
   alternates column- and row-wise standard-deviation normalization in log space,
   equalizing variance across the tile and shrinking quantization error before any
   rounding happens.

4. **Quantized Cache**: **asymmetric round-to-nearest** at low bit-width, with the
   scales folded back in at read time (keys per channel, values per token).

The shipped preset spends **more bits on keys than values** (`kvarn_k4v2_g128`:
4-bit keys, 2-bit values). We chose to release this configuration because it meets
the strictest accuracy bar, matching FP16, that the most demanding production
deployments and vLLM require, while still delivering throughput above FP16.

---

## Citation

KVarN is the official vLLM implementation of our paper:

> 📄 *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation
> in Reasoning Tasks* ([arXiv:2606.03458](https://arxiv.org/abs/2606.03458))

If you use KVarN, please cite:

```bibtex
@misc{muller2026kvarn,
      title={KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks}, 
      author={Lorenz K. Muller and Philippe Bich and Chiara Boretti and Hyun-Min Chang and Jiawei Zhuang and Lukas Cavigelli},
      year={2026},
      eprint={2606.03458},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={http://arxiv.org/abs/2606.03458}
}
```

---

## License and attribution

KVarN is built on [vLLM](https://github.com/vllm-project/vllm) (v0.28.0) and is
released under the Apache 2.0 License. The original vLLM README is preserved as
[`README_vLLM.md`](README_vLLM.md).
