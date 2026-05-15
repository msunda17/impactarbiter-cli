"""Tests for the RadixAttention SymPy oracle."""

import pytest

from src.oracles import radix_oracle
from src.fuzzer import RADIX_TEST_MATRIX


@pytest.mark.parametrize("case", RADIX_TEST_MATRIX, ids=lambda c: c.note)
def test_radix_oracle_matches_expected_matrix(case):
    lb, off = radix_oracle(case.b_local_idx, case.prefix_length, 0, 16)
    assert (lb, off) == (case.expected_block, case.expected_offset)


def test_radix_zero_case_clean_boundary():
    # prefix=48, b=0 → Agent_B_Start = 48 % 16 = 0 → executor at offset 0 of next block.
    assert radix_oracle(0, 48, 0, 16) == (3, 0)


def test_radix_partial_block_carry_over():
    # prefix=47, b=0 → carry-over into the partial block.
    assert radix_oracle(0, 47, 0, 16) == (2, 15)
