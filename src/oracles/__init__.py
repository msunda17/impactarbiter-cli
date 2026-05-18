"""SymPy-backed ground-truth oracles.

Each oracle module exposes:
    - A symbolic expression (sympy AST) capturing the ground truth.
    - A lambdified callable usable from the PyTorch autograd trap.
"""

from .paged_attention import paged_oracle
from .radix_attention import radix_oracle
from .radix_2d import radix_2d_oracle

__all__ = [
    "paged_oracle",
    "radix_oracle",
    "radix_2d_oracle",
]
