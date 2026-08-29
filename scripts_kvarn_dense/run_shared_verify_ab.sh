#!/bin/bash
# G2 serving A/B for the shared-dequant MTP verify path.
#
# Single-process A/B: boots the model ONCE (enforce_eager) and runs two
# passes over 8 fixed prompts (8K/32K/131K), toggling KVARN_SHARED_VERIFY
# between them (0 = per-token fallback, 1 = shared-dequant, the new default).
# Booting once makes the model / GEMM tactics / allocator state identical
# across both passes, so the verify kernel is the ONLY thing that differs --
# immune to the boot-to-boot non-determinism (independent fp4 autotune
# caches, CUDA-graph pool layouts) that made the two-process design's
# off-vs-on divergence unattributable.
#
# PASS requires:
#   - 8/8 prompt token-id sequences identical between the passes
#   - no asserts / IMA / CUDA errors in the log
#   - the MTP drafter actually came up (the verify path is what's under test)
#
# Prereq: stop llama-server / any other vLLM on the GPU (needs full 32 GB).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=.venv/bin/python

echo "=== G2 single-process A/B (off then on, eager) ==="
OUT_OFF=/tmp/svab_off.json OUT_ON=/tmp/svab_on.json \
  $PY scripts_kvarn_dense/shared_verify_serve_ab.py \
  > /tmp/svab.log 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "Engine core initialization failed" /tmp/svab.log; then
  echo "(engine init failed — retrying once after 20s)"
  sleep 20
  OUT_OFF=/tmp/svab_off.json OUT_ON=/tmp/svab_on.json \
    $PY scripts_kvarn_dense/shared_verify_serve_ab.py \
    > /tmp/svab.log 2>&1
  rc=$?
fi
if [ $rc -ne 0 ]; then
  echo "FAIL: run crashed (rc=$rc)"; tail -40 /tmp/svab.log; exit 1
fi

$PY - <<'EOF'
import json
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


off = load('/tmp/svab_off.json')
on = load('/tmp/svab_on.json')

bad = []
for a, b in zip(off, on):
    ta, tb = a['output_token_ids'], b['output_token_ids']
    if ta == tb:
        continue
    k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
             min(len(ta), len(tb)))
    print(f'DIVERGENCE prompt {a["idx"]}: first at step {k}')
    print('  off:', ta[max(0, k - 5):k + 20])
    print('  on :', tb[max(0, k - 5):k + 20])
    print(f'  lens off={len(ta)} on={len(tb)}')
    bad.append(a['idx'])

with open('/tmp/svab.log', errors='replace') as f:
    log = f.read()

asserts = []
for pat in ('AssertionError', 'index out of bounds',
            'illegal memory access', 'CUDA error'):
    if pat in log:
        asserts.append(pat)

mtp_up = 'drafter ready' in log

ok = True
if bad:
    print(f'FAIL: {len(bad)} prompts diverged: {bad}')
    ok = False
if asserts:
    print(f'FAIL: asserts/errors in log: {asserts}')
    ok = False
if not mtp_up:
    print('FAIL: MTP drafter did not come up (verify path not exercised)')
    ok = False
if not bad:
    print('8/8 prompts token-identical (off vs on)')
print('G2 SERVE A/B (single-process):', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
EOF
