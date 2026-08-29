# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving A/B for the shared-dequant MTP verify path (G2 gate).

Single-process A/B: boots the production model (Qwen3.8-27B NVFP4 + KVarN
kv cache + MTP+2) ONCE with enforce_eager=True, then runs two passes over
the same 8 deterministic prompts (token-padded to ~8K / ~32K / ~131K),
toggling KVARN_SHARED_VERIFY between them (0 = per-token fallback,
1 = shared-dequant, the new default). enforce_eager makes the verify
kernel launch eagerly so the env-var toggle actually changes the kernel;
booting once makes the model, GEMM tactics, and allocator state identical
across both passes, so the verify kernel is the ONLY thing that differs.
(Two separate boots were tried first: they are not bit-reproducible against
themselves — independent fp4 autotune caches and CUDA-graph pool layouts
differ per process — so off-vs-on divergence could not be attributed to
the verify kernel.)

Dumps per-prompt output token ids + text to $OUT_OFF and $OUT_ON. The
runner requires the token-id sequences to be identical across the two
passes.

Env: MODEL (default the production NVFP4 checkpoint), KV, OUT_OFF,
OUT_ON (both required), GPU.
"""

import json
import os

from vllm import LLM, SamplingParams

# (idx, target prompt tokens, item to ask about; item <= target//14 always)
TARGETS = [
    (0, 8192, 42),
    (1, 8192, 77),
    (2, 32768, 13),
    (3, 32768, 91),
    (4, 32768, 5),
    (5, 131072, 33),
    (6, 131072, 64),
    (7, 131072, 8),
]


def prompt_for(target_tokens: int, item: int) -> str:
    """Fixed reference-list prompt: ~14 tokens per item sentence, so
    target//14 items lands the prompt near the target length. The list
    is deterministic, so greedy output is reproducible across runs."""
    n_items = max(target_tokens // 14, 10)
    items = " ".join(
        f"Item {i}: codename {chr(65 + i % 26)}{i}, value {(i * 7) % 101}."
        for i in range(1, n_items + 1)
    )
    return (
        "Reference document (use it for every question):\n"
        + items
        + f"\n\nQuestion: what is the value of item {item}? "
        "Answer with the single number only."
    )


def main():
    model = os.environ.get("MODEL", "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090")
    kv = os.environ.get("KV", "kvarn_k4v2_g128")
    out_off = os.environ["OUT_OFF"]
    out_on = os.environ["OUT_ON"]

    kw = dict(
        model=model,
        quantization="modelopt",
        dtype="bfloat16",
        kv_cache_dtype=kv,
        block_size=128,
        max_model_len=262144,
        # c1: the deployment/audit regime. Must stay 1: with 4-way
        # concurrency, chunked-prefill total_k (3x131K) exceeds the FA
        # materialize buffer and the fp32 slow path OOMs the card.
        max_num_seqs=1,
        # Auto-sized KV pool at 0.90, NOT the production 9.55 GB pin: the pin
        # sits at the fork's 0.96 capacity edge and needs the card fully free.
        # Token identity is unaffected by pool size.
        gpu_memory_utilization=0.90,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
        trust_remote_code=True,
        # Eager: the verify kernel must launch eagerly so the
        # KVARN_SHARED_VERIFY toggle below actually changes which kernel runs
        # (under CUDA-graph capture the launch is frozen at capture time).
        enforce_eager=True,
    )

    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    prompts = [prompt_for(t, item) for _, t, item in TARGETS]
    print("[svab] boot done; running single-process A/B (off then on)",
          flush=True)

    def run_pass(tag, val):
        os.environ["KVARN_SHARED_VERIFY"] = val
        outs = llm.generate(prompts, sp)
        rows = []
        for (idx, target, _), o in zip(TARGETS, outs):
            out0 = o.outputs[0]
            rows.append(
                dict(
                    idx=idx,
                    target_tokens=target,
                    n_prompt_tokens=len(o.prompt_token_ids),
                    output_token_ids=list(out0.token_ids),
                    text=out0.text,
                )
            )
            print(f"[svab:{tag}] prompt {idx}: in={len(o.prompt_token_ids)} "
                  f"out={len(out0.token_ids)} text={out0.text[:60]!r}",
                  flush=True)
        return rows

    saved = os.environ.get("KVARN_SHARED_VERIFY")
    try:
        rows_off = run_pass("off", "0")
        with open(out_off, "w") as f:
            json.dump(rows_off, f, indent=2)
        rows_on = run_pass("on", "1")
        with open(out_on, "w") as f:
            json.dump(rows_on, f, indent=2)
    finally:
        if saved is None:
            os.environ.pop("KVARN_SHARED_VERIFY", None)
        else:
            os.environ["KVARN_SHARED_VERIFY"] = saved

    print("saved:", out_off, out_on, flush=True)


if __name__ == "__main__":
    main()
