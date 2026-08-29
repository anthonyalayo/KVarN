#!/usr/bin/env python
"""Single-process KVarN verify-attention microbench SWEEP (confound-free).

Drives ``kvarn_verify_attention`` (the MTP+2 verify path) directly on synthetic
int4 KV data - no server, no model weights, no MTP draft - so the measurement
is immune to the two confounds that invalidate the two-boot serving A/B:
  * boot-to-boot non-determinism (independent fp4 autotune caches, graph pools),
  * acceptance-rate variance on random prompts (which entangles serving TPOT
    with how many draft tokens each boot happens to accept).

Sweeps context lengths in ONE process (default 16k/32k/64k/128k). Each context
is set up fresh (its own KV / block-table / seq-lens), measured, then freed, so
the result is the flag's per-step verify-attention effect vs context: ~0 at
16k (weight/GEMM-bound) growing toward the 3x->1x KV-read saving at 128k. The
shared-dequant path (KVARN_SHARED_VERIFY=1, the new default) reads each block's
KV once for all QLEN tokens instead of once per token, so it must be <= the
per-token fallback (KVARN_SHARED_VERIFY=0).

Usage (needs a free GPU, ~1 min per context):
  .venv/bin/python results/kvarn_audit/verify_microbench.py
  VMB_C=16384,32768,65536,131072 .venv/bin/python results/kvarn_audit/verify_microbench.py
  VMB_OUT=results/kvarn_audit/vmb_sweep.json ...   (optional JSON out)

Output: per context, shared vs per-token median ms + the delta
(negative = shared faster), as a curve.
"""

import json
import math
import os

import torch

# Full-attention head geometry for this deployment.
HQ, HK, D = 24, 4, 256
GROUP = 128                    # == block size (g128 preset)
QLEN = 3                       # MTP+2: 2 drafts + 1 target
B = 1                          # c1 verify step
MAX_MODEL_LEN = 262144


def fill_scales(kv, cfg, dev, rng):
    """Write plausible fp16 per-channel/per-row scales into the int4 tiles."""
    Ddim, G = cfg.head_dim, cfg.group
    n = kv.shape[0]

    def draw(shape):
        return (0.001 + 0.049 * torch.rand(shape, device=dev,
                                           dtype=torch.float32,
                                           generator=rng)).to(torch.float16)

    def zdraw(shape):
        return (0.049 * (torch.rand(shape, device=dev, dtype=torch.float32,
                                    generator=rng) - 0.5)).to(torch.float16)

    def put(off, vec):
        kv[:, :, off:off + 2 * vec.shape[-1]] = (
            vec.contiguous().view(torch.uint8).view(n, HK, -1))

    put(cfg.k_s_col_offset, draw((n, HK, Ddim)))
    put(cfg.k_zp_offset, zdraw((n, HK, Ddim)))
    put(cfg.k_s_row_offset, draw((n, HK, G)))
    put(cfg.v_s_col_offset, draw((n, HK, Ddim)))
    put(cfg.v_s_row_offset, draw((n, HK, G)))
    put(cfg.v_zp_offset, zdraw((n, HK, G)))


def measure_call(fn, iters):
    """Median/p10/p90 GPU wall-time (ms) of repeated ``fn()`` calls."""
    for _ in range(3):            # warm + autotune
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {"med_ms": round(ms[len(ms) // 2], 4),
            "p10_ms": round(ms[max(0, len(ms) // 10)], 4),
            "p90_ms": round(ms[min(len(ms) - 1, 9 * len(ms) // 10)], 4)}


def setup_ctx(C, impl, cfg, tile, dev, rng):
    """Build fresh verify inputs for context length C (steady-state int4)."""
    n_blocks = math.ceil(C / GROUP)
    kv = torch.randint(0, 255, (n_blocks, HK, tile), dtype=torch.uint8,
                       device=dev, generator=rng)
    fill_scales(kv, cfg, dev, rng)
    impl._block_to_slot_t.fill_(-1)          # every block int4, no tail pool

    max_blocks_per_req = math.ceil(MAX_MODEL_LEN / GROUP)
    bt = torch.full((B, max_blocks_per_req), -1, dtype=torch.int32, device=dev)
    bt[0, :n_blocks] = torch.arange(n_blocks, dtype=torch.int32, device=dev)

    committed = C - QLEN                      # cached tokens before the QLEN
    seq_lens = torch.tensor([C], dtype=torch.int32, device=dev)
    NQ = B * QLEN
    vq_req = torch.div(torch.arange(NQ, device=dev, dtype=torch.int32), QLEN,
                       rounding_mode="floor")
    vq_seqlen = torch.tensor([committed + j + 1 for j in range(QLEN)],
                             dtype=torch.int32, device=dev)
    q = (torch.randn(NQ, HQ, D, device=dev, generator=rng) * 0.1).to(torch.float16)
    max_ctx_blocks = math.ceil(C / GROUP)
    return dict(kv=kv, bt=bt, seq_lens=seq_lens, vq_req=vq_req,
                vq_seqlen=vq_seqlen, q=q, max_ctx_blocks=max_ctx_blocks,
                n_blocks=n_blocks)


def main() -> None:
    Cs = [int(x) for x in os.environ.get(
        "VMB_C", "16384,32768,65536,131072").split(",") if x.strip()]
    out_path = os.environ.get("VMB_OUT", "")
    iters = int(os.environ.get("VMB_ITERS", "40"))

    from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
    from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionImpl
    from vllm.v1.attention.ops.triton_kvarn_decode import kvarn_verify_attention

    dev = torch.device("cuda")
    torch.cuda.init()

    cfg = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", D)
    assert cfg.group == GROUP
    tile = cfg.tile_bytes_aligned
    rng = torch.Generator(device=dev).manual_seed(20260829)

    impl = KVarNAttentionImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HK, kv_cache_dtype="kvarn_k4v2_g128")
    impl._max_model_len = MAX_MODEL_LEN
    impl._ensure_pool(dev, num_blocks_hint=math.ceil(max(Cs) / GROUP) + 128)

    saved = os.environ.get("KVARN_SHARED_VERIFY")
    rows = []
    try:
        for C in Cs:
            s = setup_ctx(C, impl, cfg, tile, dev, rng)
            results = {}
            for path, val in (("shared", "1"), ("pertok", "0")):
                os.environ["KVARN_SHARED_VERIFY"] = val
                r = measure_call(
                    lambda: kvarn_verify_attention(
                        s["q"], s["kv"], s["bt"], impl.scale, cfg, impl,
                        s["vq_req"], s["vq_seqlen"], s["max_ctx_blocks"],
                        qlen=QLEN, seq_lens=s["seq_lens"]), iters)
                results[path] = r
            d = results["shared"]["med_ms"] - results["pertok"]["med_ms"]
            rows.append({"c": C, "n_blocks": s["n_blocks"], **results,
                         "delta_ms": round(d, 4)})
            print(f"[vmb] C={C:>7} shared={results['shared']['med_ms']:.3f}ms "
                  f"pertok={results['pertok']['med_ms']:.3f}ms "
                  f"delta={d:+.3f}ms", flush=True)
            del s                            # free this context's KV
            torch.cuda.empty_cache()
    finally:
        if saved is None:
            os.environ.pop("KVARN_SHARED_VERIFY", None)
        else:
            os.environ["KVARN_SHARED_VERIFY"] = saved
    torch.cuda.synchronize()

    print()
    print(f"{'ctx':>7} | {'shared':>8} {'pertok':>8} {'delta_ms':>9} {'delta%':>8} | n_blocks")
    print("-" * 60)
    for row in rows:
        s = row["shared"]["med_ms"]
        p = row["pertok"]["med_ms"]
        d = row["delta_ms"]
        pct = (100.0 * d / p) if p else 0.0
        print(f"{row['c']:>7} | {s:8.3f} {p:8.3f} {d:+9.3f} {pct:+8.1f} | {row['n_blocks']}")
    print()
    print("delta = shared - pertok (ms, ONE layer's verify-attention call per step;")
    print("negative = shared faster). Grows with context as the 3x->1x KV-read saving")
    print("takes over the weight/GEMM-bound floor. The full-model verify step is this")
    print("delta summed over the attention layers, so serving TPOT recovery scales with")
    print("layer count x delta (and is capped by the weight/GEMM floor at short ctx).")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({"qlen": QLEN, "iters": iters, "rows": rows}, fh, indent=1)
        print(f"[vmb] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
