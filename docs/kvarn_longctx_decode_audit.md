# KVarN long-context decode audit — Qwen3.8-27B-NVFP4 on a single RTX 5090

Date: 2026-08-28 · Status: complete (no code changes; findings + recommendations)
Artifacts: `results/kvarn_audit/` (bench logs, nsys rep/sqlite, microbench JSONs, scripts)

## Questions

1. Is the measured c1 TPOT shape — flat at 8K/32K (~10 ms), step at 131K (~20 ms),
   plateau at 245K (~20 ms) — as expected?
2. Is the KVarN/vLLM implementation doing something naive?
3. Where does the lag come from, and can it be improved?

## TL;DR

- **No plateau.** Clean re-runs: 131K = 15.7 ms, 196K = 19.8 ms, 245K = 21.9 ms
  (median TPOT, c1, MTP+2). The old "20/20" rows were rounding plus one suspect 245K row
  (tokenizer warning `265238 > 262144`, not reproduced in the clean run: all 16/16 prompts
  succeeded, `max_model_len = 262144`). The apparent 131K "step" is real but it is the
  KV-read cost crossing over the context-independent weight-streaming GEMM cost — a regime
  change, not a cliff.
- **Yes, one naive part.** With MTP+2, the target-model **verify pass re-reads the entire
  KV context once per verify token**: 3 tokens × 17 full-attention layers × 134 MB
  (131K) = **6.8 GB of KV traffic per step, of which ~4.5 GB is redundant**. The fix — the
  shared-dequant verify kernel (`_kvarn_fused_verify_stage1`, all QLEN tokens share each
  block's dequant, KV bytes = single-token decode) — **exists and is numerically validated,
  but is default-OFF** (`KVARN_SHARED_VERIFY=0`, `triton_kvarn_decode.py:1066-1074`)
  because serving with it corrupts MTP drafter proposals through a not-yet-isolated
  mechanism (suspected async-scheduling/drafter-metadata interaction, not kernel math).
- **The step size is accounted for exactly**: q_len=3 verify graph = 27.3 ms; q_len=1
  (MTP-off) step = 16.0 ms; Δ = 11.2 ms ≈ 2 × 17 layers × 0.331 ms (measured
  per-layer 1-token KV read) = 11.3 ms.
- **Top-ROI improvement**: isolate and fix the shared-verify corruption → ~10–12 ms/step
  recovered at 131K (TPOT 15.7 → ~10.7 ms, back to the 8K/32K floor; 245K 21.9 → ~16.5).
  Autotune-space and split-count changes (the original plan's items 1–2) are worth ≤1 ms/step
  — not worth doing (microbench below).
- Secondary, minor: int4 fused decode kernel runs at ~405 GB/s = **22.6% of DRAM peak**
  (consistent with the code's own ncu note: L1/TEX-transaction-bound at ~25% occupancy);
  Q_PER_KV 6 is padded to 8 (25% of Q-side work masked). Even at 100% DRAM, a 1× read at
  131K is 1.3 ms/step — kernel efficiency is not the dominant term.

## Deployment & model facts (verified on disk)

- GPU: RTX 5090, 32,607 MiB, **170 SMs**, 96 MiB L2, ~1.79 TB/s DRAM (GDDR7).
- Model: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (modelopt NVFP4). HF config
  (`text_config`): 64 layers, `full_attention_interval=4` → **16 full-attention layers**
  + 48 linear-attention (fixed-size state) + **1 MTP full-attention layer** = **17 KVarN
  paged layers**. Hq=24, Hk=4, D=256 (GQA 6:1 → padded 8:1 in-kernel).
  `max_position_embeddings = 262144`; `--max-model-len auto` → 262144.
- KVarN preset `kvarn_k4v2_g128`: key 4-bit, value 2-bit, **group = 128 = vLLM block size**
  (the preset requires `block_size == group`; the earlier plan note "block=64, 17,920 B
  tile" was from a D=128/k4v4 deployment — corrected here).
  Tile per (block, head): k_packed 16,384 + k_scale 1,408 + v_packed 8,192 + v_scale 1,024
  = 27,008 B → aligned slot 256 B/token → **32,768 B per (block, head)**
  (`KVarNConfig.tile_bytes_aligned`).
  **Per-token KV: 17 layers × 4 heads × 256 B = 17,408 B ≈ 17.0 KiB/token**
  (1,024 B per layer-token; 134.2 MB/layer at 131K, 251.7 MB/layer at 245K).
- Boot (pinned, from `ctx_sweep.sh`): `--quantization modelopt --kv-cache-dtype
  kvarn_k4v2_g128 --max-model-len auto --max-num-seqs 4 --kv-cache-memory 9551856271
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (+ qwen3 chat
  template/parsers). KV pool capacity per boot log: ~490,822 tokens ≈ 3,834 blocks × 128.
- Boot log also warns: untuned fp4 GEMM fallback at verify-batch shapes (M=2,4,5,10) —
  the known c4 throughput cliff; separate issue, out of scope here.

## Method & artifacts (all under `results/kvarn_audit/`)

|Step |Artifact |Result|
|---|---|---|
|nsys session trace of one 131K c1 request (MTP+2, warmup excluded)|`nsys_131k.nsys-rep` / `.sqlite`, `nsys_131k_kern_sum.txt`|per-kernel + per-CUDA-graph breakdown|
|131K c1 ×16 (MTP+2 and MTP-off)|`bench_131k_c1_mtp.log`, `bench_131k_c1_mtpoff.log`|TPOT 15.72 / 16.07 ms median|
|245K c1 ×16, 196,608 c1 ×8|`bench_245k_c1_mtp.log`, `bench_196k_c1_mtp.log`|TPOT 21.90 / 19.81 ms median|
|Standalone microbench: `kvarn_decode_attention` at C ∈ {32K,131K,245K} × splits {64,128,256} × {fused split-K, fused single-stage, materialize} (synthetic random int4 KV, production autotune state)|`mb_*.json`, `mb_*.log`, `microbench.py`|table below|
|ncu on stage1 @131K|`run_phase2c.sh` step 7 (not run in this window; command fixed and ready)|see Reproducing|

Bench: `vllm bench serve --dataset-name random --random-range-ratio 0.0 --random-output-len
128 --max-concurrency 1 --request-rate inf --ignore-eos`. Note: median **TTFT** in these
logs is inflated by client-side queueing (all prompts submitted at t=0 behind the c1
semaphore: median TTFT ≈ prefill + (rank−1)×per-request time); actual 131K prefill is
2.7 s from the nsys trace.

## Findings

### 1. TPOT rows (clean, this run)

|ctx (input)|c1 TPOT median, MTP+2|c1 TPOT median, MTP-off|Δ vs 32K|
|---|---|---|---|
|8,192 / 32,768 (prior sweep)|10 / 10|—|—|
|131,072|**15.72**|16.07|+5.7|
|196,608|19.81|—|+9.8|
|245,760|21.90|—|+11.9|

Monotonic, roughly linear above 32K (slope ≈ 45–55 ms per 1M tokens ≈ effective KV-read
rate ~300–460 GB/s). No plateau; no discrete cliff at 131K in the clean data. MTP+2 ≈
MTP-off in per-token latency at 131K (15.7 vs 16.1): the extra verify work is paid for by
accepting ~2.3–3.6 tokens/step.

### 2. Per-step structure at 131K (nsys, two independent traces agree)

One MTP iteration ≈ 31–57 ms wall (varies with draft acceptance on random input),
comprised of:

|Phase|Time|Notes|
|---|---|---|
|Target verify forward (64 layers, q_len=3, captured CUDA graph)|**median 27.3 ms** (n=36; other trace 27.6 ms, n=48)|graph-internal kernels not individually reported by this nsys config — only graph-level records|
|MTP draft forwards (eager): stage1 (1,4,64) 348 μs + LM-head GEMM ~650 μs ×2 + scoring/sampling|~4 ms|visible per-kernel|
|Small piecewise graphs + bookkeeping|~0.6 ms|×2–4 per iteration|

TPOT = step wall ÷ tokens/step (2.3–3.6) ≈ 15–17 ms/token, matching the bench.

### 3. The KV read — per-layer cost, live vs microbench

Live (nsys, MTP layer = one of the 17 full-attention layers, 131K context):

|call|grid|median|bytes|effective BW|
|---|---|---|---|---|
|MTP draft (q=1)| (1,4,64) |348 μs|134.2 MB|386 GB/s (21.5%)|
|MTP **verify (q=3)**| (3,4,64) |**871 μs** (2.50× draft)|134.2 MB|154 GB/s per program-row, 462 GB/s aggregate|

Microbench (`microbench.py`, one layer, same autotuned config the server uses):

|C|split-K 64 (production)|split-K 128|split-K 256|single-stage|materialize|
|---|---|---|---|---|---|
|32,768 (34 MB)|**0.083 ms**|0.106|0.128|2.511|0.407|
|131,072 (134 MB)|**0.331 ms**|0.356|0.366|12.681|1.667|
|245,760 (252 MB)|**0.610 ms**|0.632|0.558|24.130|3.155|

- 0.331 ms microbench vs 0.348 ms live on the identical grid → the microbench faithfully
  reproduces server kernel behavior.
- 134.2 MB / 0.331 ms = **405 GB/s = 22.6% of the 1.79 TB/s DRAM peak**. This is the
  "KV read runs far below peak" anomaly from the plan; it matches the code's own ncu note
  (fused kernel register-limited to ~25% occupancy, L1/TEX-transaction-bound, not
  DRAM-bandwidth-bound). ncu re-verification not run in this window (command in
  Reproducing); not needed for the verdict — see item 4.
- 64 splits is the sweet spot at 131K; 256 is only marginally better at 245K
  (−0.05 ms/layer ≈ −0.9 ms/step across 17 layers) and worse at 131K. Raising
  `KVARN_MAX_KV_SPLITS` is **not** worth a code change; the existing `KVARN_NUM_KV_SPLITS`
  env override covers 245K-class deployments without any change.
- Single-stage (no split-K) is 38× slower at 131K (4 programs on 170 SMs) — split-K
  flash-decoding is doing its job; materialize is 5× slower (build_packed 1.32 ms + FA
  0.34 ms).

### 4. Root cause of the context growth: the 3× verify re-read

`kvarn_verify_attention` (`triton_kvarn_decode.py:1010`) has two modes:

- **UNIFORM shared-dequant** (`_kvarn_fused_verify_stage1`): one program per
  (request, kv-head, split); the request's QLEN verify tokens **share** each block's
  dequant → KV bytes + dequant ALU equal single-token decode.
- **Per-token fallback** (`_kvarn_fused_decode_stage1` with `VQ_INDIRECT`): grid
  `(NQ, Hk, SPLITS)` — each verify token independently walks the whole context.

The shared-dequant mode is gated behind `KVARN_SHARED_VERIFY` with **default "0"**
(lines 1066–1074): it was numerically validated in isolation (matches the per-token kernel
within fp32 reduction noise on live inputs), but "serving with it corrupts the MTP
drafter's proposals (invalid [-1,...] spec tokens, embedding index asserts at
temperature>0, degenerate greedy output) through a mechanism not yet isolated — suspicion
is an interaction with async scheduling / drafter metadata rather than kernel math.
Re-enable for debugging only."

Consequences per step at 131K (MTP+2, 3 verify tokens, 17 full-attention layers):

|mode|KV traffic/step|attention time/step (at measured ~460 GB/s aggregate)|
|---|---|---|
|per-token (production today)|17 × 3 × 134.2 MB = **6.83 GB**|~14.8 ms|
|shared-dequant (fix)|17 × 134.2 MB = **2.28 GB**|~5.0 ms|

Cross-checks, all consistent:

- MTP-off (1× reads) 131K step = 16.0 ms; MTP+2 verify graph = 27.3 ms;
  **Δ = 11.2 ms ≈ 2 × 17 × 0.331 ms = 11.3 ms** (the two extra token-walks).
- Live MTP-layer verify/draft ratio = 2.50× on identical KV bytes.
- Slope 131K→245K: predicted Δ per step = 17 × (1.525 − 0.871) ms ≈ 11.1 ms →
  ΔTPOT ≈ 11.1 / tokens-per-step ≈ 5–7 ms; measured 6.2 ms.
- At 8K/32K the re-read is small (0.42 / 1.7 GB per step), so TPOT sits on the
  ~10 ms weight-streaming floor and the step is invisible — explaining the flat low end.

### 5. Answers

1. **As expected?** The flat 8K/32K floor is expected (weight-streaming bound: 27B NVFP4
   weights ≈ 14 GB → ~8–10 ms at DRAM peak, plus overhead). The 131K step is real but is a
   regime change (KV-read cost overtaking GEMM cost), amplified by the 3× verify re-read.
   The 245K "plateau" does not exist in clean data — it was rounding plus a suspect row.
2. **Naive part?** Yes, exactly one significant: the per-token verify walk
   (QLEN× redundant context reads + dequant) is the production path because the correct
   shared-dequant path is default-off pending a serving-bug isolation. Minor, known,
   accepted: Q_PER_KV 6→8 padding (25% masked Q rows); int4 kernel at ~22.6% of DRAM peak
   (L1/TEX-bound, occupancy-limited — real but secondary, see below).
3. **Can it be improved?** Yes — the only big lever is the shared-dequant verify.
   Even at 100% DRAM bandwidth a 1× read at 131K costs 1.3 ms/step; today's 14.8 ms is
   dominated by the redundant reads, so kernel-efficiency work (autotune/splits/occupancy)
   caps out at ≤1 ms/step and does not touch the dominant term.

## Recommendations (ROI order)

1. **Isolate and fix the `KVARN_SHARED_VERIFY` corruption** (report-only here; no code
   changed in this audit). This is the ~10–12 ms/step win at ≥131K
   (131K TPOT 15.7 → ~10.7 ms ≈ the 8K/32K floor; 245K 21.9 → ~16.5 ms; −73% per-step KV
   traffic at 131K). Starting points: the kernel math is already validated in isolation, so
   instrument the *serving* side — diff MTP drafter inputs/metadata (spec token buffer,
   `vq_req`/`vq_seqlen` plans, accepted-token bookkeeping) between
   `KVARN_SHARED_VERIFY=0/1` at the first divergence; suspect async scheduling / drafter
   metadata per the code comment. Gate: `scripts_kvarn_dense/run_validation.sh` + the
   cosine/IMA checks, plus a spec-decode acceptance run (look for `[-1,...]` spec tokens).
2. **No autotune-space or split-count change.** Microbench shows 64 splits optimal at
   131K; 256 splits buy ~0.9 ms/step only at 245K and is already reachable via the existing
   `KVARN_NUM_KV_SPLITS` env (no code change) if a 245K-class deployment wants it.
3. **c4 cliff** (20→280 ms at c4, untuned fp4 GEMM fallback at M=2,4,5,10 per boot log):
   separate known issue, separate FlashInfer tuning pass — out of scope here.
4. Optional, if per-byte kernel efficiency becomes the bottleneck after #1: run the ncu
   command below; the code's standing note (L1/TEX-bound at ~25% occupancy) is the
   hypothesis to confirm/refute, and wider `BLOCK_N`/`num_warps=8`/`maxnreg` points are the
   natural candidates.

## Reproducing

```bash
cd /home/anthony/Desktop/KVarN

# Bench rows (needs free GPU; ~5 min each):
bash results/kvarn_audit/bench_c1.sh 33331 131072 16 <tag>   # server must be up on :33331
# server boot: results/kvarn_audit/boot_nsys.sh (MTP+2) / boot_mtp_off.sh (control)

# Microbench (needs free GPU; ~2-3 min per C):
MB_C=131072 MB_OUT=results/kvarn_audit/mb_131072.json .venv/bin/python results/kvarn_audit/microbench.py

# ncu on the stage1 kernel at 131K (pinned production config, 8 launches, needs free GPU):
MB_NCU=1 MB_C=131072 /usr/local/cuda/bin/ncu --profile-from-start off \
  -k regex:stage1 --launch-count 8 \
  --section SpeedOfLight --section MemoryWorkloadAnalysis \
  --section Occupancy --section LaunchStats --section WarpStateStats \
  -f -o results/kvarn_audit/ncu_stage1_131k \
  .venv/bin/python results/kvarn_audit/microbench.py
/usr/local/cuda/bin/ncu --import results/kvarn_audit/ncu_stage1_131k.ncu-rep --page details

# Optional: decompose the verify graph with --enforce-eager (all kernels visible):
bash results/kvarn_audit/run_phase2c.sh   # boots eager server, traces, microbench, ncu
```

## Caveats

- Acceptance rate on random-token prompts varies run to run (2.3–3.6 tokens/step
  observed), so per-step wall time and TPOT jitter by ±20%; per-step GPU work (27.3 ms
  graph) is stable across both traces.
- nsys in this configuration reports CUDA-graph kernels only as graph-level records
  (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`), so the 27.3 ms verify graph's internal per-kernel
  split is inferred (per-layer attention from live MTP-layer measurements + microbench,
  GEMMs weight-streaming-bound) rather than directly summed. `run_phase2c.sh`
  (`--enforce-eager`) would produce the direct per-kernel breakdown if desired.
- Microbench KV data is synthetic random uint8 (layout-valid); steady-state
  `block_to_slot = -1` (all-int4) regime — the same regime as long decode.

## Resolution (2026-08-29): shared-dequant verify enabled by default

The fix for finding #4 / recommendation #1 is implemented and gated. This
section documents the root cause, the change, and the verification gates.

### Root cause: the shared-dequant path could not compile for this deployment

The "corruption" the old `DEFAULT OFF` comment attributed to an unisolated
serving mechanism was, for this model, a **structural compile failure masked by
two fallbacks**:

1. **Q-tile rows are not a power of 2 for MTP+2.** The kernel builds its Q tile
   as `M = QLEN * Q_PER_KV_PAD` rows. Production geometry: QLEN = 3 (MTP+2),
   GQA 6:1 padded to 8:1 -> `Q_PER_KV_PAD = 8` -> **M = 24**. Triton's
   `tl.arange(0, M)` requires a power of 2 (verified in this venv, triton 3.7.1:
   `arange's range must be a power of 2`), so the kernel cannot compile at
   M = 24. The driver gate enforced this (`(_m & (_m - 1)) == 0`) and silently
   fell back to the per-token path.
2. **The warmup never compiled it anyway.** `_warm_decode_kernels` sourced QLEN
   from the global `get_current_vllm_config()` — a contextmanager that is only
   set during the engine's forward, *not* when `_ensure_pool` first runs (dummy-
   run profiling / first metadata build). It read `_qlen = 0`, so the verify
   kernel's JIT + autotune sweep was deferred to the first live verify step,
   inside CUDA-graph capture. Any failure there surfaced mid-serve as the
   drafter "corruption" the comment described, not at boot.
3. **Default off.** `KVARN_SHARED_VERIFY` defaulted to "0" (belt-and-braces on
   top of 1+2).

QLEN = 2 (MTP+1) gives M = 16 and QLEN = 4 (MTP+3) gives M = 32 — both
naturally valid, which is why the bug was invisible outside the MTP+2
deployment.

### The fix (3 edits, `vllm/v1/attention/`)

**Kernel** (`ops/triton_kvarn_decode.py`, `_kvarn_fused_verify_stage1`):
add a `Q_TILE_ROWS: tl.constexpr` parameter and build the tile over
`tl.arange(0, Q_TILE_ROWS)` with
`Q_TILE_ROWS = 1 << max(4, (QLEN*Q_PER_KV_PAD - 1).bit_length())` — a power of
2, at least 16 (tl.dot's M floor). The row mask becomes
`rmask = (lane < Q_PER_KV) & (j < QLEN)`; padded rows load `q = 0` (masked
load) and are masked out of every store. The causal limit, score/acc temporaries,
and stage-2 combine are unchanged in math — padded rows simply carry no data.
Cost: MTP+2 tiles are 32 rows instead of 24 (25% extra **Q-side** work, masked;
KV bytes + dequant remain 1x — the point of the fix).

**Driver** (`kvarn_verify_attention`): drop the `(_m & (_m - 1)) == 0` gate
condition and flip the default to ON —
`os.environ.get("KVARN_SHARED_VERIFY", "1") == "1"`; `KVARN_SHARED_VERIFY=0`
reverts to the per-token fallback.

**Warmup** (`backends/kvarn_attn.py`):
- The builder derives `self._spec_qlen = 1 + num_speculative_tokens` from its
  own `vllm_config.speculative_config` in `__init__` (always available) and
  passes it through `build()` -> `_ensure_pool(verify_qlen=...)`.
- The verify kernel now warms under its **own key**
  `("verify", device, qlen, ...)` — the decode key is consumed by the impl-side
  dummy-run `_ensure_pool` calls (which carry `verify_qlen=0`) before the
  builder's first real call, so a shared key would mask the verify warmup.
- The warmup launches `_kvarn_fused_verify_stage1` with the deployment's real
  QLEN/Q_TILE_ROWS at boot, so the autotune sweep runs before capture. Compile
  errors propagate on purpose: a broken verify kernel fails boot loudly instead
  of mid-serve.

### G1 gate correction (2026-08-29)

The original G1 unit A/B had a state-construction bug: it set
`block_to_slot = 0` for every block (`zero_()`), which routes **every** block
through the fp16 tail-pool branch and silently **skips the int4 dequant path**
entirely — yet production uses int4 for all full blocks. G1 passing was
therefore over-optimistic. Fixed to `fill_(-1)` + per-request tail slots (the
production convention: full blocks int4, only the in-progress tail in the pool).
A standalone probe of the corrected setup confirms the int4 path is within float
noise of the per-token path (max_abs ~1.5e-05 / ~2e-07 depending on case), i.e.
the int4 dequant is **not** the source of any serving divergence.

### Verification gates

Master runner: `scripts_kvarn_dense/run_shared_verify_gates.sh` (needs a free
GPU; `g3` arg re-runs only the perf round). Unit A/B:
`scripts_kvarn_dense/shared_verify_unit_ab.py`; serving A/B:
`scripts_kvarn_dense/shared_verify_serve_ab.py` + `run_shared_verify_ab.sh`.

|Gate|Method|Result|
|---|---|---|
|G0|pre-commit + import, no new violations vs baseline|PASS (2026-08-29)|
|G1|Unit A/B: shared vs per-token on identical synthetic int4 + tail-pool state, seeded; 6 cases incl. the deployment geometry (144/24 heads, QLEN=3), the 64K x B=7 shape, the 8K floor, MTP+1/+3 shape classes, and the no-GQA dot floor; now covers the int4 path (G1 b2s fix above); PASS = finite + within fp32 noise|PASS — 6/6 (full gate run 2026-08-28)|
|G2|**Single-process** serving A/B: one enforce_eager boot, two passes over 8 greedy prompts (8K/32K/131K) toggling `KVARN_SHARED_VERIFY=0/1`; the verify kernel is then the ONLY thing that differs (immune to the boot-to-boot non-determinism that invalidated the two-process version); PASS = 8/8 token-identical + MTP drafter up + no IMA/asserts|PASS — 8/8 token-identical (2026-08-28)|
|G3|Perf: **single-process verify microbench** (`results/kvarn_audit/verify_microbench.py`) driving `kvarn_verify_attention` directly at c1 across 16K/32K/64K/128K **in one process** (fresh KV per context, freed between), `KVARN_SHARED_VERIFY=0` vs default; PASS = shared <= per-token per-step latency at every context. (The two-boot serving A/B was retired as a perf gate: its TPOT is confounded by boot-to-boot acceptance variance on random prompts, not KV traffic)|PASS — shared faster at every context (2026-08-29): 16K -0.062 ms (-47%), 32K -0.117 ms (-48%), 64K -0.137 ms (-36%), 128K -0.281 ms (-34%); absolute saving grows with context|

**Why G2 changed from two boots to one process — and the outcome:** the first
G2 run (two separate boots) diverged on 2/8 prompts, one to a wrong answer
(item 42 -> 70 instead of 92). Investigation showed the two boots are not
bit-reproducible against themselves — independent fp4 autotune caches (105 vs
104 configs, different cache hashes) and per-process CUDA-graph pool layouts —
so an off-vs-on difference could not be attributed to the verify kernel. The
reworked single-process A/B (one eager boot, verify kernel toggled between
passes) resolved it: **8/8 prompts token-identical**, confirming the earlier
divergence was boot non-determinism (draft-model + autotune + graph-pool
state), not the shared-dequant math.

G2's token identity is the correctness gate. G3 (single-process verify
microbench) confirms the perf win at the kernel level across the context
sweep - one process, fresh KV per context, no MTP, so acceptance variance is
impossible by construction:

    ctx   shared  pertok  delta    delta%
   16K   0.069   0.131   -0.062   -47%
   32K   0.128   0.245   -0.117   -48%
   64K   0.241   0.378   -0.137   -36%
  128K   0.536   0.817   -0.281   -34%

The shared path is faster at every context (-34% to -48% on the verify
attention step), and the absolute saving grows with context (0.062 ->
0.281 ms) - the 3x->1x KV-read model. The two-boot serving matrix
(16K/32K/64K/128K) is NOT a clean perf gate: on random prompts the MTP
acceptance rate is a noisy, window-dependent quantity (it swung 1.68-2.09
within a single arm, and the two arms disagreed at 32K/64K), so its TPOT gap
was an acceptance artifact - except at 128K, where acceptance was ~equal
(1.73 vs 1.71) and the flag showed its real end-to-end effect (-3.27 ms/
token, 34% faster). The microbench is the valid perf gate; the 128K serving
row is the realized end-to-end TPOT, and together they tell the full story
(clean mechanism + end-to-end recovery, the latter capped by the weight/GEMM
floor).

### Remaining known issues (unchanged by this fix)

- **c4 throughput cliff** (untuned fp4-GEMM fallback at verify-batch shapes
  M = 2,4,5,10): separate known issue, out of scope.
- Padded Q tiles cost 25% masked Q-side rows at QLEN = 3 (and QLEN = 5 -> 64
  rows, QLEN = 6 -> 64); negligible next to the 3x->1x KV-traffic change.
- MTP+1/MTP+3 geometries (M = 16/32) were never broken; they now warm and run
  through the same default-ON path.
