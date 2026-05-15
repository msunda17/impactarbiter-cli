"""Tests for the PagedAttention SymPy oracle."""

import pytest

from src.oracles import paged_oracle
from src.fuzzer import PAGED_ADVERSARIAL_TOKENS


@pytest.mark.parametrize(
    "tok,bs,expected",
    [
        (0, 16, (0, 0)),
        (15, 16, (0, 15)),
        (16, 16, (1, 0)),
        (99, 16, (6, 3)),
        (100, 16, (6, 4)),
        (105, 16, (6, 9)),
        (128, 16, (8, 0)),
    ],
)
def test_paged_oracle_known_values(tok, bs, expected):
    assert paged_oracle(tok, bs) == expected


def test_paged_oracle_consistency_over_adversarial_tokens():
    for tok in PAGED_ADVERSARIAL_TOKENS:
        lb, off = paged_oracle(tok, 16)
        assert lb * 16 + off == tok
