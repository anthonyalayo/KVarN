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
        KVarNConfig.from_cache_dtype("kvarn_k4v2_g128", 128).tile_bytes_aligned // 128
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


def test_sw_skip_layer_block_fills_shared_page():
    """Real-model regression (Qwen3.8-27B KVarN + dflash2): the draft's SWA
    layers are skip layers (dense bf16, 4096 B/token) padded to the shared
    3,407,872-byte page. FlashAttention advertises MultipleOf(16) — any
    multiple of 16 is a valid kernel block — so the largest block whose page
    fits the shared page is 832 (fills it exactly), not the base 16. At
    block 16 the 6,143-token window (2047 + 4096 in-flight) spans
    cdiv(6143, 16) + 1 = 385 near-empty 3.4 MB blocks per request and
    auto-fit collapses the pool from the full 262144 context to 43264
    tokens; at block 832 it is 9 blocks and the full context fits."""
    from vllm.model_executor.layers.attention.attention import (
        _largest_kernel_block_within,
    )
    from vllm.utils.math_utils import cdiv
    from vllm.v1.attention.backend import MultipleOf

    class _MultipleOf16Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(16)]

    block = _largest_kernel_block_within(_MultipleOf16Backend, 4096, 3_407_872, 128)
    assert block == 832
    assert cdiv(6143, block) + 1 == 9


def test_sw_kernel_block_no_budget_uses_smallest_supported():
    """Without a page budget (no shared page to pad into) the smallest
    supported block is returned — unify scales it up by an integer ratio."""
    from vllm.model_executor.layers.attention.attention import (
        _largest_kernel_block_within,
    )
    from vllm.v1.attention.backend import MultipleOf

    class _MultipleOf16Backend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [MultipleOf(16)]

    assert _largest_kernel_block_within(_MultipleOf16Backend, 4096, None, 128) == 16
    assert _largest_kernel_block_within(_MultipleOf16Backend, 0, 3_407_872, 128) == 16


def test_sw_kernel_block_fixed_sizes_unchanged():
    """Backends advertising fixed int sizes keep the existing behavior: the
    largest listed size that fits the budget, else the smallest."""
    from vllm.model_executor.layers.attention.attention import (
        _largest_kernel_block_within,
    )

    class _FixedBackend:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [64, 128]

    # 128 * 108 = 13824 fits 3_407_872; budget smaller than the 128 page
    # falls back to 64.
    assert _largest_kernel_block_within(_FixedBackend, 108, 3_407_872, 128) == 128
    assert _largest_kernel_block_within(_FixedBackend, 108, 8192, 128) == 64
    # Nothing fits: the smallest listed size is returned (pre-existing).
    assert _largest_kernel_block_within(_FixedBackend, 108, 1024, 128) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
