# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for ``KVarNMetadataBuilder._tile_row_table`` — the
expansion of a vLLM-block block table to KVarN tile rows.

Hybrid models inflate the unified attention block above the KVarN tile
size (e.g. 3200 = 25 x 128), so the kernels' block-table rows and the
vLLM block table are different unit systems. The helper is the ONLY
place they meet; a wrong expansion means unguarded kernel loads (IMA)
or attention over the wrong physical KV rows. Pure numpy — no torch,
GPU, or model required.
"""

import numpy as np
import pytest

from vllm.v1.attention.backends.kvarn_attn import KVarNMetadataBuilder

GROUP = 128  # KVarN tile size (kvarn_k4v2_g128)


def tile_table(row, seq_len, tpr, max_tiles, num_rows=1):
    """Build a 1-request block table and expand it via the helper."""
    block_table_np = np.full((num_rows, len(row)), -1, dtype=np.int32)
    block_table_np[:, : len(row)] = row
    return KVarNMetadataBuilder._tile_row_table(
        block_table_np, [seq_len] * num_rows, tpr, GROUP, max_tiles)


def test_tpr1_is_identity_regression_guard():
    """tpr=1 (non-hybrid: block size == tile size): the expanded table must
    be the raw table verbatim, -1-padded — hybrid inflation must be a no-op
    on models whose block size never grew."""
    row = [5, 9, -1]
    out = tile_table(row, seq_len=3 * GROUP, tpr=1, max_tiles=8)
    expected = np.array([5, 9, -1, -1, -1, -1, -1, -1], dtype=np.int32)
    np.testing.assert_array_equal(out[0], expected)


def test_tpr25_two_full_vllm_blocks():
    """tpr=25, row [7, 12, -1], seq_len = 2 full vLLM blocks (6400 tokens =
    50 tiles): tiles 0..24 expand to block 7's rows, 25..49 to block 12's."""
    out = tile_table([7, 12, -1], seq_len=2 * 25 * GROUP, tpr=25, max_tiles=64)
    np.testing.assert_array_equal(out[0, :25], 7 * 25 + np.arange(25))
    np.testing.assert_array_equal(out[0, 25:50], 12 * 25 + np.arange(25))
    assert (out[0, 50:] == -1).all()


def test_seq_len_exactly_one_vllm_block():
    """seq_len=3200 = exactly one 3200-token vLLM block = 25 tiles: only
    the first 25 tile rows are valid."""
    out = tile_table([7], seq_len=25 * GROUP, tpr=25, max_tiles=32)
    np.testing.assert_array_equal(out[0, :25], 7 * 25 + np.arange(25))
    assert (out[0, 25:] == -1).all()


def test_seq_len_exactly_one_tile():
    """seq_len=128: exactly one tile. Tile 0 -> 7*25; tiles >=1 -> -1."""
    out = tile_table([7], seq_len=GROUP, tpr=25, max_tiles=32)
    assert out[0, 0] == 7 * 25
    assert (out[0, 1:] == -1).all()


def test_seq_len_crossing_one_vllm_block_boundary():
    """seq_len=3328 = one full vLLM block + 128 tokens into the next: tile
    25 is the first tile of block 12; the rest is padding."""
    out = tile_table([7, 12], seq_len=25 * GROUP + GROUP, tpr=25, max_tiles=32)
    np.testing.assert_array_equal(out[0, :25], 7 * 25 + np.arange(25))
    assert out[0, 25] == 12 * 25
    assert (out[0, 26:] == -1).all()


def test_unallocated_first_block_yields_all_minus_one():
    """Row starting with -1 (unallocated first vLLM block): every entry is
    -1 — the kernels skip -1 rows, so no pool slot or int4 load is attempted."""
    out = tile_table([-1, 12], seq_len=2 * 25 * GROUP, tpr=25, max_tiles=16)
    assert (out == -1).all()


def test_seq_len_zero_yields_all_minus_one():
    """seq_len=0: no tiles, every entry -1."""
    out = tile_table([7, 12], seq_len=0, tpr=25, max_tiles=16)
    assert (out == -1).all()


def test_multi_row_and_short_table_width():
    """Two requests with different lengths; the table is narrower than
    the full max_tiles width. Row 0 spans both vLLM blocks (6400 tokens),
    row 1 only the first (3200)."""
    out = KVarNMetadataBuilder._tile_row_table(
        np.array([[7, 12], [7, -1]], dtype=np.int32),
        [2 * 25 * GROUP, 25 * GROUP], tpr=25, group=GROUP, max_tiles=64)
    np.testing.assert_array_equal(out[0, :25], 7 * 25 + np.arange(25))
    np.testing.assert_array_equal(out[0, 25:50], 12 * 25 + np.arange(25))
    assert (out[0, 50:] == -1).all()
    np.testing.assert_array_equal(out[1, :25], 7 * 25 + np.arange(25))
    assert (out[1, 25:] == -1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
