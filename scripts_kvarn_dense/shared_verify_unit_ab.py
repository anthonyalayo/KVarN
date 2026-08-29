#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit A/B for the shared-dequant MTP verify kernel (G1 gate).

Compares the shared-dequant uniform verify path
(``_kvarn_fused_verify_stage1``, ``KVARN_SHARED_VERIFY=1``, ``qlen=QLEN``)
against the per-token VQ_INDIRECT fallback (``qlen=0``) on IDENTICAL
synthetic int4 + tail-pool state. Covers the deployment geometry
(Qwen3.8-27B: 144/24 heads, D=256, MTP+2 -> QLEN=3, M=24 -> 32 padded
tile rows), the June-corruption shape class (M already a power of 2),
and the no-GQA tl.dot floor (M=2 -> 16 rows).

Each request's LAST block is pointed at a tail-pool slot (``pool_slot
>= 0`` branch) with the rest in the int4 cache; ragged committed
lengths leave that final tile partially filled (``cmask`` branch).
Scale vectors are drawn from a small range (not raw random bytes) so
the dequantized KV stays inside fp16 range and the PASS condition
``isfinite(out_shared)`` is meaningful.

Deterministic (seeded), no model weights, ~3.5 GB VRAM peak per case
(sized to run while a llama-server occupies the rest of the GPU).
Exit 0 iff every case PASSes.
"""

import math
import os
import sys

import torch

GROUP = 128  # == vllm block size (g128 preset)

# (HQ, HK, D, QLEN, C, B, label)
CASES = [
    (144, 24, 256, 3, 131072, 1, "deploy-131K-B1"),
    (144, 24, 256, 3, 131072, 3, "deploy-131K-B3"),
    (144, 24, 256, 3, 65536, 7, "deploy-64K-B7"),
    (144, 24, 256, 3, 8192, 3, "deploy-8K-B3"),
    (64, 16, 256, 2, 65536, 3, "corrupt-class-M8"),
    (16, 16, 256, 2, 8192, 2, "noGQA-dotfloor-M16"),
]


def fill_scales(kv, cfg, HK, dev, rng):
    """Write small per-block/per-channel scale+zeropoint vectors into the
    tile's fp16 fields (the random bytes in the packed K/V regions stay).
    Without this, raw fp16 bytes dequantize to O(1e6) values and the fp16
    outputs overflow to inf."""
    D, G = cfg.head_dim, cfg.group
    n = kv.shape[0]

    def draw(shape):
        return (
            0.001
            + 0.049 * torch.rand(shape, device=dev, dtype=torch.float32, generator=rng)
        ).to(torch.float16)

    def zdraw(shape):
        return (
            0.049
            * (torch.rand(shape, device=dev, dtype=torch.float32, generator=rng) - 0.5)
        ).to(torch.float16)

    def put(off, vec):
        kv[:, :, off : off + 2 * vec.shape[-1]] = (
            vec.contiguous().view(torch.uint8).view(n, HK, -1)
        )

    put(cfg.k_s_col_offset, draw((n, HK, D)))
    put(cfg.k_zp_offset, zdraw((n, HK, D)))
    put(cfg.k_s_row_offset, draw((n, HK, G)))
    put(cfg.v_s_col_offset, draw((n, HK, D)))
    put(cfg.v_s_row_offset, draw((n, HK, G)))
    put(cfg.v_zp_offset, zdraw((n, HK, G)))


def make_impl(HQ, HK, dev):
    from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
    from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionImpl

    D = 256
    cfg = KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", D)
    assert cfg.group == GROUP
    # Standalone _ensure_pool scratch sizing must fit the ~4 GB GPU window
    # (llama-server still up): the FA packed scratch (materialize-fallback
    # only — kvarn_verify_attention never touches it) is sized
    # fa_rows = max(min(_max_num_seqs * _max_model_len, 262144),
    #               _max_model_len, 4096) -> 4096 rows here. The kernel's
    # split count comes from the driver's max_ctx_blocks argument, not from
    # these; KVARN_POOL_SLOTS (set in main) caps the tail pool at 64 slots.
    impl = KVarNAttentionImpl(
        num_heads=HQ, head_size=D, scale=1.0 / math.sqrt(D),
        num_kv_heads=HK, kv_cache_dtype="kvarn_k4v2_g128")
    impl._max_model_len = 4096
    impl._max_num_seqs = 1
    impl._max_num_batched_tokens = 2048
    return impl, cfg, D


def run_case(HQ, HK, QLEN, C, B, dev):
    """One A/B case: build state, run shared vs per-token, return metrics."""
    from vllm.v1.attention.ops.triton_kvarn_decode import kvarn_verify_attention

    D = 256
    impl, cfg, _ = make_impl(HQ, HK, dev)
    rng = torch.Generator(device=dev).manual_seed(20260829)

    # Varying committed lengths break uniformity (ragged block counts).
    committed = [C - (b * 1024 + 13 * b) for b in range(B)]
    n_blocks = [math.ceil((committed[b] + QLEN) / GROUP) for b in range(B)]
    total = sum(n_blocks)
    tile = cfg.tile_bytes_aligned

    kv = torch.randint(
        0, 255, (total, HK, tile), dtype=torch.uint8, device=dev, generator=rng
    )
    fill_scales(kv, cfg, HK, dev, rng)
    impl._ensure_pool(dev, num_blocks_hint=total + 8)

    # Unique physical blocks per request row.
    bt = torch.full((B, max(n_blocks)), -1, dtype=torch.int32, device=dev)
    bid = 0
    for b in range(B):
        bt[b, : n_blocks[b]] = torch.arange(
            bid, bid + n_blocks[b], dtype=torch.int32, device=dev
        )
        bid += n_blocks[b]

    # All blocks -> int4 cache (b2s = -1), EXCEPT each request's LAST block,
    # which is a partial tile and lives in the tail pool (b2s = slot >= 0).
    # This matches production: full blocks take the int4 dequant path, only the
    # in-progress tail takes the fp16 pool path. (Was zero_()=0, which routed
    # EVERY block to the pool path and silently skipped the int4 dequant.)
    impl._block_to_slot_t.fill_(-1)
    for b in range(B):
        last = int(bt[b, n_blocks[b] - 1])
        impl._block_to_slot_t[last] = b
        impl._tail_K_pool[b] = (
            torch.randn(GROUP, HK, D, device=dev, generator=rng) * 0.01
        ).to(torch.float16)
        impl._tail_V_pool[b] = (
            torch.randn(GROUP, HK, D, device=dev, generator=rng) * 0.01
        ).to(torch.float16)

    # vq plan, the builder's formula: token b*QLEN+j attends
    # committed_b + j + 1 keys; the request's full length is the LAST entry.
    NQ = B * QLEN
    vq_req = torch.div(
        torch.arange(NQ, device=dev, dtype=torch.int32), QLEN, rounding_mode="floor"
    )
    vq_seqlen = torch.zeros(NQ, dtype=torch.int32, device=dev)
    seq_lens = torch.zeros(B, dtype=torch.int32, device=dev)
    for b in range(B):
        vq_seqlen[b * QLEN : (b + 1) * QLEN] = torch.tensor(
            [committed[b] + j + 1 for j in range(QLEN)], dtype=torch.int32, device=dev
        )
        seq_lens[b] = committed[b] + QLEN

    q = (torch.randn(NQ, HQ, D, device=dev, generator=rng) * 0.1).to(torch.float16)

    # Per-case context bound -> the deployment's split schedule for C
    # (adaptive: 64 splits above 256 blocks, 32 at or below).
    max_ctx_blocks = math.ceil(C / GROUP)

    saved = os.environ.get("KVARN_SHARED_VERIFY")
    try:
        os.environ["KVARN_SHARED_VERIFY"] = "1"
        out_shared = kvarn_verify_attention(
            q,
            kv,
            bt,
            impl.scale,
            cfg,
            impl,
            vq_req,
            vq_seqlen,
            max_ctx_blocks,
            qlen=QLEN,
            seq_lens=seq_lens,
        )
        out_pertok = kvarn_verify_attention(
            q, kv, bt, impl.scale, cfg, impl, vq_req, vq_seqlen, max_ctx_blocks, qlen=0
        )
    finally:
        if saved is None:
            os.environ.pop("KVARN_SHARED_VERIFY", None)
        else:
            os.environ["KVARN_SHARED_VERIFY"] = saved
    torch.accelerator.synchronize()

    finite = bool(torch.isfinite(out_shared).all())
    max_abs = float((out_shared - out_pertok).abs().max())
    diff_l2 = float((out_shared - out_pertok).norm())
    ref_l2 = float(out_pertok.norm())
    rel_l2 = diff_l2 / ref_l2 if ref_l2 > 0 else float("inf")
    del kv, bt, q, out_shared, out_pertok
    return dict(finite=finite, max_abs=max_abs, rel_l2=rel_l2)


def main() -> None:
    dev = torch.device("cuda")
    os.environ["KVARN_POOL_SLOTS"] = "64"

    rows = []
    for HQ, HK, D, QLEN, C, B, label in CASES:
        res = run_case(HQ, HK, QLEN, C, B, dev)
        ok = res["finite"] and res["max_abs"] < 5e-3 and res["rel_l2"] < 1e-2
        rows.append((label, res["max_abs"], res["rel_l2"], ok))
        torch.accelerator.empty_cache()

    print(f"{'case':<24} {'max_abs':>10} {'rel_l2':>10}  verdict")
    for label, ma, rl, ok in rows:
        print(f"{label:<24} {ma:>10.3e} {rl:>10.3e}  {'PASS' if ok else 'FAIL'}")
    failed = [r for r in rows if not r[3]]
    print(f"shared-verify unit A/B: {len(rows) - len(failed)}/{len(rows)} PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
