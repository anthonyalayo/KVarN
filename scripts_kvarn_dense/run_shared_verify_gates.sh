#!/bin/bash
# Master gate runner for the shared-dequant MTP verify fix: G1 -> G2 -> G3.
#
#   bash scripts_kvarn_dense/run_shared_verify_gates.sh
#
# PREREQ: stop llama-server / any other process on the GPU — the gates need
# the full 32 GB (G1 alone would fit in ~4.7 GB, G2/G3 won't).
# Prints PASS/FAIL per gate and STOPS at the first FAIL (a G2 divergence must
# not proceed to G3; see the contingency notes in the audit resolution).
# Logs: /tmp/svab_*.log (G1/G2), /tmp/g3_serve_*.log +
# results/kvarn_audit/bench_g3_*.log (G3). ~40 min total.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=.venv/bin/python
PORT=8000

# ── sanity: GPU must be free ─────────────────────────────────────────────────
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "ABORT: another process holds the GPU (stop it first):"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  exit 1
fi
echo "GPU free — starting gates $(date)"

GATE="${1:-all}"   # all (default) | g3 = re-run only the perf round
if [[ "$GATE" != "g3" ]]; then

# ── G1: unit A/B (shared vs per-token on synthetic state) ──────────────────
echo "=== G1 unit A/B ==="
$PY scripts_kvarn_dense/shared_verify_unit_ab.py | tee /tmp/svab_g1.log
g1=${PIPESTATUS[0]}
if [ $g1 -ne 0 ]; then
  echo "GATE SUMMARY: G1 FAIL — stop here (kernel mismatch on synthetic state)."
  exit 1
fi
echo "G1 PASS"

# ── G2: serving A/B (single-process eager, greedy token identity) ─────────
echo "=== G2 serving A/B (single-process eager, ~15 min) ==="
bash scripts_kvarn_dense/run_shared_verify_ab.sh
g2=$?
if [ $g2 -ne 0 ]; then
  echo "GATE SUMMARY: G2 FAIL — STOP (do not run G3)."
  echo "  divergence data + logs: /tmp/svab_off.json /tmp/svab_on.json"
  echo "  /tmp/svab_off.log /tmp/svab_on.log"
  exit 1
fi
echo "G2 PASS"

else
  echo "(g3-only mode: skipping G1 + G2 — confirmed in a prior run)"
fi

# ── G3: perf gate (pinned sweep server, c1 131K + 8K, 16x128) ──────────────
SERVE_ARGS=(serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
  --quantization modelopt
  --chat-template /home/anthony/qwen38-froggeric-v22.jinja
  --kv-cache-dtype kvarn_k4v2_g128
  --max-model-len auto
  --max-num-seqs 4
  --kv-cache-memory 9551856271
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --limit-mm-per-prompt '{"image": 4}'
  --mm-processor-kwargs '{"max_pixels": 8388608}')

bench_round() {  # bench_round <tag> [env KEY=VAL ...]
  local tag=$1; shift
  echo "=== G3 boot: $tag ==="
  env "$@" .venv/bin/vllm "${SERVE_ARGS[@]}" \
    > "/tmp/g3_serve_${tag}.log" 2>&1 &
  local srv=$!
  local ok=0
  for _ in $(seq 1 216); do            # up to 18 min for boot
    sleep 5
    if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
      ok=1; break
    fi
    kill -0 "$srv" 2>/dev/null || break
  done
  if [ $ok -ne 1 ]; then
    echo "G3: server '$tag' never became ready (see /tmp/g3_serve_${tag}.log)"
    tail -30 "/tmp/g3_serve_${tag}.log"
    kill "$srv" 2>/dev/null
    return 1
  fi
  echo "G3: server '$tag' ready — benching 131K + 8K"
  bash results/kvarn_audit/bench_c1.sh "$PORT" 131072 16 "g3_${tag}_131k" \
    || { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; return 1; }
  bash results/kvarn_audit/bench_c1.sh "$PORT" 8192 16 "g3_${tag}_8k" \
    || { kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; return 1; }
  kill "$srv" 2>/dev/null
  wait "$srv" 2>/dev/null
  return 0
}

med() { grep -oP 'Median TPOT \(ms\):\s+\K[0-9.]+' \
          "results/kvarn_audit/bench_$1.log" | tail -1; }
acc() { grep -oP 'Acceptance rate \(%\):\s+\K[0-9.]+' \
          "results/kvarn_audit/bench_$1.log" | tail -1; }

unset KVARN_SHARED_VERIFY
bench_round off KVARN_SHARED_VERIFY=0 || {
  echo "GATE SUMMARY: G3 FAIL (off round) — see /tmp/g3_serve_off.log"; exit 1; }
bench_round on || {
  echo "GATE SUMMARY: G3 FAIL (on round) — see /tmp/g3_serve_on.log"; exit 1; }
m_off131=$(med g3_off_131k); m_on131=$(med g3_on_131k)
m_off8=$(med g3_off_8k);     m_on8=$(med g3_on_8k)
a_off131=$(acc g3_off_131k); a_on131=$(acc g3_on_131k)
echo "=== G3 numbers ==="
echo "TPOT median 131K: off=${m_off131} ms  on=${m_on131} ms"
echo "TPOT median 8K:   off=${m_off8} ms  on=${m_on8} ms"
echo "acceptance 131K:  off=${a_off131}%  on=${a_on131}%"

$PY - "$m_on131" "$m_off131" "$m_off8" "$m_on8" <<'EOF'
import sys
m_on131, m_off131, m_off8, m_on8 = (float(x) for x in sys.argv[1:5])
ok = True
r = "PASS" if m_on131 <= 12.0 else "FAIL"
print(f"G3 131K default TPOT {m_on131:.2f} ms (baseline {m_off131:.2f}) -> {r} (gate <= 12.0)")
ok &= m_on131 <= 12.0
d = abs(m_on8 - m_off8)
r = "PASS" if d <= 0.5 else "FAIL"
print(f"G3 8K delta {d:.2f} ms ({m_off8:.2f} -> {m_on8:.2f}) -> {r} (gate <= 0.5)")
ok &= d <= 0.5
sys.exit(0 if ok else 1)
EOF
g3=$?

echo "=================== GATE SUMMARY ==================="
if [[ "$GATE" == "g3" ]]; then
  echo "G1 unit A/B:    (skipped — g3-only mode)"
  echo "G2 serving A/B: (skipped — g3-only mode)"
else
  echo "G1 unit A/B:    PASS"
  echo "G2 serving A/B: PASS (8/8 token-identical, single-process eager)"
fi
echo "G3 perf:        $([ $g3 -eq 0 ] && echo PASS || echo FAIL)"
echo "  TPOT median 131K: off=${m_off131} -> on=${m_on131} ms (expected ~10.5-11.5)"
echo "  TPOT median 8K:   off=${m_off8} -> on=${m_on8} ms (must stay within 0.5)"
[ $g3 -eq 0 ] && echo "ALL GATES PASS — safe to restart llama-server." \
  || echo "G3 FAIL — keep the GPU busy; do NOT restart llama-server yet."
exit $g3
