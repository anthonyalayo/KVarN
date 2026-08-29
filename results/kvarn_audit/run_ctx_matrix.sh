#!/bin/bash
# Context-window performance matrix: shared-dequant MTP verify flag ON vs OFF.
#
# Standard `vllm bench serve` does the measurement; this wrapper only toggles
# KVARN_SHARED_VERIFY between two boots and sweeps the context sizes.
#
#   * c1, fixed 128 output tokens -> identical number of decode steps per cell
#   * same --seed for both arms    -> identical random prompts across arms
#   * --random-range-ratio defaults to 0.0 -> each row is exactly that context
#
# Boot command is the production winner, with --kv-cache-memory replaced by
# --gpu-memory-utilization 0.9 (no pinned KV budget needed for this sweep).
#
# Usage:  bash results/kvarn_audit/run_ctx_matrix.sh
# Needs a free GPU (~32 GB). Per arm: one model load + 4 bench cells.
# Output: results/kvarn_audit/ctx_matrix/ctx<C>_{on,off}.{json,log},
#         server_{on,off}.log, and a summary matrix printed at the end.
set -u
cd "$(dirname "$0")/../.."

MODEL=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
CHAT="${HOME}/qwen38-froggeric-v22.jinja"
PORT=8000
OUT_LEN=128
NUM_PROMPTS=16
CONCURRENCY=1
SEED=0
CTX_LIST=(16384 32768 65536 131072)
OUT_DIR=results/kvarn_audit/ctx_matrix
mkdir -p "$OUT_DIR"
VLLM=.venv/bin/vllm
PYBIN=.venv/bin/python

cleanup() { pkill -f "vllm serve" 2>/dev/null || true; }
trap cleanup EXIT

boot() { # $1=flag(on|off)  $2=log
  local flag=$1 log=$2
  if [ "$flag" = off ]; then export KVARN_SHARED_VERIFY=0; else unset KVARN_SHARED_VERIFY; fi
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  echo "=== boot: shared_verify=$flag ==="
  "$VLLM" serve "$MODEL" \
    --quantization modelopt --chat-template "$CHAT" \
    --kv-cache-dtype kvarn_k4v2_g128 --max-model-len auto --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --limit-mm-per-prompt '{"image": 4}' --mm-processor-kwargs '{"max_pixels": 8388608}' \
    --port "$PORT" > "$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "  ready (pid=$pid)"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "  server died during boot; log tail:"; tail -40 "$log"; return 1
    fi
    sleep 2
  done
  echo "  server not ready in 600s; log tail:"; tail -40 "$log"; return 1
}

bench() { # $1=flag  $2=ctx
  local flag=$1 ctx=$2
  echo "=== bench: ctx=$ctx shared_verify=$flag ==="
  "$VLLM" bench serve --backend vllm --model "$MODEL" \
    --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "$ctx" --random-output-len "$OUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONCURRENCY" \
    --seed "$SEED" --label "ctx${ctx}_${flag}" \
    --save-result --result-filename "$OUT_DIR/ctx${ctx}_${flag}.json" \
    > "$OUT_DIR/ctx${ctx}_${flag}.log" 2>&1
}

for flag in on off; do
  boot "$flag" "$OUT_DIR/server_${flag}.log" || { echo "boot failed for $flag"; exit 1; }
  for ctx in "${CTX_LIST[@]}"; do
    bench "$flag" "$ctx"
  done
  pkill -f "vllm serve" 2>/dev/null; sleep 5
done

echo
echo "================ SUMMARY MATRIX ================"
"$PYBIN" - "$OUT_DIR" <<'PY'
import json, os, re, sys

out = sys.argv[1]
ctxs = [16384, 32768, 65536, 131072]

def cell(flag, ctx):
    try:
        with open(f"{out}/ctx{ctx}_{flag}.json") as f:
            return json.load(f)
    except Exception:
        return {}

def get(d, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return v
    return None

def boot_acc(logpath):
    # Aggregate mean acceptance length over the whole arm (last SpecDecoding
    # metrics line). This is per-BOOT, not per-cell.
    try:
        with open(logpath, errors="replace") as f:
            txt = f.read()
    except Exception:
        return None
    m = re.findall(r"Mean acceptance length: ([\d.]+)", txt)
    return float(m[-1]) if m else None

def fmt(x, w):
    return (f"{x:.3f}" if isinstance(x, (int, float)) else "-").rjust(w)

on_acc = boot_acc(f"{out}/server_on.log")
off_acc = boot_acc(f"{out}/server_off.log")

hdr = (f"{'ctx':>7} | {'TPOT_on':>8} {'TPOT_off':>9} {'Δ on-off':>9} "
       f"| {'thr_on':>8} {'thr_off':>8} {'Δ':>6} | {'acc_on':>7} {'acc_off':>8}")
print(hdr)
print("-" * len(hdr))
for ctx in ctxs:
    o = cell("on", ctx); f = cell("off", ctx)
    tpo = get(o, "mean_tpot_ms", "tpot", "mean_tpot")
    tpf = get(f, "mean_tpot_ms", "tpot", "mean_tpot")
    tho = get(o, "output_throughput", "throughput")
    thf = get(f, "output_throughput", "throughput")
    dt = (tpo - tpf) if (tpo is not None and tpf is not None) else None
    dth = (tho - thf) if (tho is not None and thf is not None) else None
    print(f"{ctx:>7} | {fmt(tpo,8)} {fmt(tpf,9)} {fmt(dt,9)} "
          f"| {fmt(tho,8)} {fmt(thf,8)} {fmt(dth,6)} "
          f"| {fmt(on_acc,7)} {fmt(off_acc,8)}")

print()
print("TPOT = mean ms per output token (lower=better); thr = output tokens/s")
print("(higher=better); acc = mean acceptance length for the whole arm (the")
print("SpecDecoding aggregate in server_on/off.log - higher acceptance = more")
print("draft tokens reused = fewer single-token steps = lower TPOT). Same seed")
print("=> identical prompts across arms, so prompt mix is not a confound; boot")
print("autotune/graph-pool state still can nudge acc, so read acc alongside dTPOT.")
PY
