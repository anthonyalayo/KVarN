# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the v0.28 KVarN spec plumbing: the KVQuantMode kvarn
members, get_kv_quant_mode's kvarn_ branch, and KVarNAttentionBackend.
customize_spec (the 0.28 replacement for the removed TQ*Spec packed classes).
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

DENSE_DTYPES = (
    "kvarn_k4v2_g128",
    "kvarn_k4v4_g128",
    "kvarn_k4v2_g64",
    "kvarn_k4v4_g64",
)


def test_kv_quant_mode_maps_every_kvarn_dtype_string():
    """Each kvarn_ cache dtype string resolves to its KVARN mode member
    (member names mirror the dtype strings), never to NONE."""
    for dtype in (*DENSE_DTYPES, "kvarn_mla_k4_g128"):
        assert get_kv_quant_mode(dtype) is KVQuantMode[dtype.upper()]


def test_customize_spec_publishes_packed_slot_for_full_attention():
    """The dense kvarn_ full-attention spec gets state_content_bytes =
    tile_bytes_aligned // group. For kvarn_k4v2_g128 at head 128 that slot is
    108 bytes (13824-byte aligned tile / 128-token group), so a 128-token
    block page is 110,592 bytes for 8 heads."""
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.KVARN_K4V2_G128,
    )
    out = KVarNAttentionBackend.customize_spec(spec)
    expected_slot = (
        KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", 128).tile_bytes_aligned
        // 128
    )
    assert expected_slot == 108
    assert out.state_content_bytes == 108
    assert out.page_size_bytes == 128 * 8 * 108  # 110592


def test_customize_spec_reproduces_hybrid_page():
    """The real-model regression: Qwen3.8-27B runs kvarn_k4v2_g128 at
    head 256 / 4 KV heads; the slot is 256 bytes, so the 3200-token hybrid
    block page is 3,276,800 bytes per attention layer (the 0.27.1-verified
    value, now produced through the customize_spec hook)."""
    out = KVarNAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=3200,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            kv_quant_mode=KVQuantMode.KVARN_K4V2_G128,
        )
    )
    assert out.state_content_bytes == 256
    assert out.page_size_bytes == 3200 * 4 * 256  # 3_276_800


def test_customize_spec_passes_skip_layers_through():
    """Unquantized skip layers (KVQuantMode.NONE) run through the same
    backend but must keep state_content_bytes=None."""
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.NONE,
    )
    assert KVarNAttentionBackend.customize_spec(spec) is spec


def test_customize_spec_covers_sliding_window_specs():
    """KVarN sliding-window layers get plain SlidingWindowSpec in 0.28 (the
    TQSlidingWindowSpec class is gone); the hook must pack them exactly like
    the full-attention specs."""
    spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.KVARN_K4V2_G128,
        sliding_window=4096,
    )
    out = KVarNAttentionBackend.customize_spec(spec)
    assert out.state_content_bytes == 108
    assert out.page_size_bytes == 128 * 8 * 108  # 110592


def test_turboquant_mode_split_is_intact():
    """Upstream regression guard: the 0.28 TURBOQUANT -> K8V4/4BIT_NC/...
    rename did not break the dtype -> mode mapping."""
    assert KVQuantMode.TURBOQUANT_K8V4.value == 6
    assert get_kv_quant_mode("turboquant_k8v4") is KVQuantMode.TURBOQUANT_K8V4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
