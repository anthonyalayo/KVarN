#!/bin/bash
# Context-window -> throughput sweep on the single-5090 KVarN box.
# Server lifecycle is owned by the sweep (boots once, clears caches between
# bench combos). MTP+2 + kv-cache pin, the verified config.
#
# Usage:
#   bash ctx_sweep.sh --dry-run    # print the expanded commands, run nothing
#   bash ctx_sweep.sh             # full run
#   bash ctx_sweep.sh --resume    # re-run after an interruption, skip done combos
#
# Prereq: stop any other vLLM / llama-server on this GPU (the sweep binds :8000
# itself and the model needs the full 32 GB).
set -euo pipefail
cd "$(dirname "$0")"

SERVE_CMD="vllm serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --quantization modelopt \
  --chat-template /home/anthony/qwen38-froggeric-v22.jinja \
  --kv-cache-dtype kvarn_k4v2_g128 \
  --max-model-len auto \
  --max-num-seqs 4 \
  --kv-cache-memory 9551856271 \
  --speculative-config '{\"method\": \"mtp\", \"num_speculative_tokens\": 2}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --limit-mm-per-prompt '{\"image\": 4}' --mm-processor-kwargs '{\"max_pixels\": 8388608}'"

BENCH_CMD="vllm bench serve \
  --model gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --dataset-name random \
  --random-output-len 128 \
  --request-rate inf \
  --ignore-eos"

exec vllm bench sweep serve \
  --serve-cmd "$SERVE_CMD" \
  --bench-cmd "$BENCH_CMD" \
  --bench-params ctx_sweep_bench.json \
  -o results/ctx_sweep \
  -e qwen38_27b_ctx \
  --num-runs 1 \
  --server-ready-timeout 900 \
  "$@"
