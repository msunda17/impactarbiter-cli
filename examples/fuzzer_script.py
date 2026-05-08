"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ImpactArbiter — PagedAttention Block-Mapping Fuzzer                ║
║          SymPy Oracle  ×  PyTorch Differentiable Proxy  ×  Grad Audit       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  1. Oracle        — SymPy AST compiled via lambdify → logical_block, offset
  2. Proxy         — PyTorch continuous KV-cache slice + backward pass
  3. Fuzzer Loop   — 20 synthetic Agent functions (30 % correct / 70 % hallucinated)
  4. Verdict       — torch.allclose gradient comparison with FP32 tolerance
"""

import random
import sys
import math

import sympy
import torch

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
WHITE   = "\033[97m"
BG_RED  = "\033[41m"
BG_GREEN= "\033[42m"
DIM     = "\033[2m"

def banner(text: str, colour: str = CYAN) -> None:
    width = 78
    print(f"\n{colour}{BOLD}{'═' * width}{RESET}")
    print(f"{colour}{BOLD}  {text}{RESET}")
    print(f"{colour}{BOLD}{'═' * width}{RESET}\n")

def section(text: str) -> None:
    print(f"\n{MAGENTA}{BOLD}── {text} {'─' * (72 - len(text))}{RESET}")

def info(text: str) -> None:
    print(f"  {CYAN}▸{RESET}  {text}")

def ok(text: str) -> None:
    print(f"  {GREEN}{BOLD}✔{RESET}  {text}")

def warn(text: str) -> None:
    print(f"  {YELLOW}{BOLD}⚠{RESET}  {text}")

def gradient_divergence_alert(agent_id: int, hallucination: str) -> None:
    """Massive, hard-to-miss terminal alert for gradient divergence."""
    width = 78
    lines = [
        "",
        f"{BG_RED}{WHITE}{BOLD}{'█' * width}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'⚡  GRADIENT DIVERGENCE DETECTED: Memory Pointer Overlap  ⚡':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{f'  Agent ID  : #{agent_id:<4}':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{f'  Fault     : {hallucination}':<76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'  KV-Cache gradient mismatch exceeds FP32 tolerances':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'  ➜ AI-generated block mapping is MATHEMATICALLY INVALID':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█':1}{'':^76}{'█':1}{RESET}",
        f"{BG_RED}{WHITE}{BOLD}{'█' * width}{RESET}",
        "",
    ]
    print("\n".join(lines))

def pass_alert(agent_id: int) -> None:
    width = 78
    print(
        f"  {BG_GREEN}{WHITE}{BOLD}  ✔  Agent #{agent_id:<3} — "
        f"Gradient tensors MATCH Oracle  (rtol=1e-05 | atol=1e-08)  "
        f"{'':>{width - 65}}{RESET}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ORACLE — SymPy compilation
# ─────────────────────────────────────────────────────────────────────────────

def build_oracle():
    """
    Compile the deterministic block-mapping equations into a PyTorch callable.

    Returns
    -------
    oracle_fn : callable
        oracle_fn(token_idx_tensor, block_size_tensor) → (logical_block_tensor, offset_tensor)
        All operations are differentiable-free (integer floor / mod).
    """
    token_idx  = sympy.Symbol("token_idx",  nonnegative=True, integer=True)
    block_size = sympy.Symbol("block_size", positive=True,    integer=True)

    logical_block_expr = sympy.floor(token_idx / block_size)
    offset_expr        = token_idx % block_size

    # Compile to a PyTorch callable.
    # "pytorch" string alias is not supported in all SymPy releases, so we
    # hand a namespace dict that maps the SymPy primitives we actually use
    # (floor, Mod) to their PyTorch equivalents.  All other names fall back
    # to the standard math module, which is fine because the oracle only
    # performs integer floor-division and modulo — no tensor internals leak
    # into the SymPy layer.
    torch_ns = {
        "floor": lambda x: int(math.floor(x)) if not isinstance(x, torch.Tensor) else torch.floor(x).long(),
        "Mod":   lambda a, b: a % b,
        "ITE":   lambda c, t, f: t if c else f,
    }
    oracle_fn = sympy.lambdify(
        (token_idx, block_size),
        (logical_block_expr, offset_expr),
        modules=[torch_ns, "math"],
    )
    return oracle_fn


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXECUTION ENVIRONMENT — PyTorch setup
# ─────────────────────────────────────────────────────────────────────────────

# Hyper-parameters
NUM_PHYSICAL_BLOCKS = 32
BLOCK_SIZE          = 16
HEAD_DIM            = 64
NUM_KV_HEADS        = 8

torch.manual_seed(42)
random.seed(42)

# 1-D integer block_table: logical_block → physical_block_id
block_table: torch.Tensor = torch.randint(
    low=0, high=NUM_PHYSICAL_BLOCKS, size=(NUM_PHYSICAL_BLOCKS,), dtype=torch.long
)

# 3-D KV_cache: [physical_block, tokens_per_block, head_dim * num_kv_heads]
# requires_grad=True → leaf tensor for autograd
KV_CACHE_HEAD_STRIDE = HEAD_DIM * NUM_KV_HEADS
KV_cache: torch.Tensor = torch.randn(
    NUM_PHYSICAL_BLOCKS, BLOCK_SIZE, KV_CACHE_HEAD_STRIDE,
    dtype=torch.float32,
    requires_grad=True,
)

# Dummy query vector (not a leaf; we won't backprop through it)
dummy_query: torch.Tensor = torch.randn(KV_CACHE_HEAD_STRIDE, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DIFFERENTIABLE PROXY — bypass discrete indexing
# ─────────────────────────────────────────────────────────────────────────────

def run_proxy(logical_block_int: int, offset_int: int) -> torch.Tensor:
    """
    Given pre-computed discrete indices (logical_block, offset), perform:
      1. block_table look-up  →  physical_block  (discrete; no grad)
      2. Slice KV_cache       →  fetched_vector  (continuous; has grad_fn)
      3. mock_score = Σ(fetched_vector * dummy_query)
      4. mock_score.backward()
      5. Return KV_cache.grad (cloned for comparison)

    The KV_cache leaf tensor is reset between calls.
    """
    # Zero-out any accumulated gradient from a previous pass
    if KV_cache.grad is not None:
        KV_cache.grad.zero_()

    # Discrete look-up — kept outside autograd to avoid integer-backward crash
    physical_block: int = int(block_table[logical_block_int].item())

    # Continuous slice — this is where autograd attaches
    fetched_vector: torch.Tensor = KV_cache[physical_block, offset_int, :]  # shape: [HEAD_DIM * NUM_KV_HEADS]

    # Differentiable scalar via dot product with dummy query
    mock_score: torch.Tensor = torch.sum(fetched_vector * dummy_query)

    # Backward through the continuous subgraph only
    mock_score.backward()

    return KV_cache.grad.clone()


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORACLE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

ORACLE_FN = build_oracle()

def oracle_indices(token_idx: int) -> tuple[int, int]:
    """Run the compiled SymPy oracle and return (logical_block, offset) as ints."""
    lb_raw, off_raw = ORACLE_FN(
        torch.tensor(token_idx, dtype=torch.float32),
        torch.tensor(BLOCK_SIZE, dtype=torch.float32),
    )
    # lambdify returns torch tensors; convert to int for discrete indexing
    lb  = int(lb_raw.item())  if isinstance(lb_raw,  torch.Tensor) else int(lb_raw)
    off = int(off_raw.item()) if isinstance(off_raw, torch.Tensor) else int(off_raw)
    return lb, off


def oracle_grad(token_idx: int) -> torch.Tensor:
    lb, off = oracle_indices(token_idx)
    return run_proxy(lb, off)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SYNTHETIC AGENT FUNCTIONS
#    30 % correct  |  70 % mathematically hallucinated
# ─────────────────────────────────────────────────────────────────────────────

AgentResult = tuple[int, int]   # (logical_block, offset)


# ── Correct implementation ────────────────────────────────────────────────────
def _correct(token_idx: int, block_size: int) -> AgentResult:
    """Ground-truth: matches the SymPy oracle exactly."""
    return token_idx // block_size, token_idx % block_size


# ── Hallucinated implementations ──────────────────────────────────────────────
def _swap_mod_with_div(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses integer division where modulo is required."""
    logical = token_idx // block_size
    offset  = token_idx // block_size   # ← wrong: should be % block_size
    return logical, offset % block_size  # clamp to avoid index OOB


def _wrong_block_index(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: off-by-one in logical block (common LLM mistake)."""
    logical = (token_idx // block_size) + 1   # ← wrong: +1 shifts every block
    offset  = token_idx % block_size
    # Clamp so we don't blow the block_table bounds
    logical = min(logical, NUM_PHYSICAL_BLOCKS - 1)
    return logical, offset


def _inverted_mapping(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: swaps logical and offset roles entirely."""
    logical = token_idx % block_size          # ← these are swapped
    offset  = token_idx // block_size
    offset  = offset % block_size             # clamp
    return logical, offset


def _multiply_instead_of_divide(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: multiplies instead of divides for logical block."""
    logical = (token_idx * block_size) % NUM_PHYSICAL_BLOCKS  # ← wrong
    offset  = token_idx % block_size
    return logical, offset


def _wrong_base_offset(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: adds a spurious constant offset to every block."""
    logical = (token_idx // block_size) + 3   # ← wrong: constant bias
    offset  = token_idx % block_size
    logical = logical % NUM_PHYSICAL_BLOCKS
    return logical, offset


def _subtraction_bug(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses subtraction instead of modulo for offset."""
    logical = token_idx // block_size
    offset  = token_idx - logical             # ← semantically broken
    offset  = abs(offset) % block_size        # clamp
    return logical, offset


def _floor_as_ceiling(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses math.ceil instead of floor for logical block."""
    logical = math.ceil(token_idx / block_size)   # ← wrong: ceil ≠ floor
    offset  = token_idx % block_size
    logical = min(logical, NUM_PHYSICAL_BLOCKS - 1)
    return logical, offset


def _double_block_size(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses 2×block_size, under-segments the cache."""
    logical = token_idx // (block_size * 2)    # ← wrong granularity
    offset  = token_idx % block_size
    return logical, offset


def _modulo_wrong_operand(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: takes modulo of block_size by token_idx."""
    logical = token_idx // block_size
    offset  = block_size % (token_idx + 1)     # ← operands reversed
    offset  = offset % block_size
    return logical, offset


def _zero_offset_always(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: always returns offset=0 (ignores intra-block position)."""
    logical = token_idx // block_size
    offset  = 0                                # ← always wrong for token_idx > 0
    return logical, offset


def _block_from_hash(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses a hash-like scramble for logical block."""
    logical = ((token_idx * 31) + 7) % NUM_PHYSICAL_BLOCKS   # ← nonsense
    offset  = token_idx % block_size
    return logical, offset


def _reverse_index(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: indexes from the end of the block table."""
    logical = NUM_PHYSICAL_BLOCKS - 1 - (token_idx // block_size)  # ← backwards
    offset  = token_idx % block_size
    logical = max(logical, 0)
    return logical, offset


def _bit_shift_instead_of_divide(token_idx: int, block_size: int) -> AgentResult:
    """Hallucination: uses right-shift by 3 (=÷8) instead of ÷block_size."""
    logical = token_idx >> 3                   # ← hardcoded shift, wrong for block_size≠8
    offset  = token_idx % block_size
    logical = logical % NUM_PHYSICAL_BLOCKS
    return logical, offset


# ─────────────────────────────────────────────────────────────────────────────
# Agent registry: 20 agents  (6 correct ≈ 30%, 14 hallucinated ≈ 70%)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_REGISTRY: list[tuple[callable, str, bool]] = [
    #  (fn,                          label,                              is_correct)
    (_correct,                       "Correct: floor+mod",               True),
    (_swap_mod_with_div,             "Bug: mod→div for offset",          False),
    (_wrong_block_index,             "Bug: off-by-one logical block",    False),
    (_inverted_mapping,              "Bug: logical↔offset swapped",      False),
    (_multiply_instead_of_divide,    "Bug: mul instead of div",          False),
    (_wrong_base_offset,             "Bug: +3 constant bias on block",   False),
    (_correct,                       "Correct: floor+mod",               True),
    (_subtraction_bug,               "Bug: subtraction for offset",      False),
    (_floor_as_ceiling,              "Bug: ceil instead of floor",       False),
    (_double_block_size,             "Bug: 2×block_size granularity",    False),
    (_modulo_wrong_operand,          "Bug: reversed modulo operands",    False),
    (_correct,                       "Correct: floor+mod",               True),
    (_zero_offset_always,            "Bug: offset hardcoded 0",          False),
    (_block_from_hash,               "Bug: hash-scrambled block idx",    False),
    (_reverse_index,                 "Bug: reversed block table index",  False),
    (_bit_shift_instead_of_divide,   "Bug: right-shift instead of div",  False),
    (_wrong_base_offset,             "Bug: +3 constant bias on block",   False),
    (_correct,                       "Correct: floor+mod",               True),
    (_swap_mod_with_div,             "Bug: mod→div for offset",          False),
    (_inverted_mapping,              "Bug: logical↔offset swapped",      False),
]

assert len(AGENT_REGISTRY) == 20, "Registry must contain exactly 20 agents."


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUZZING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    banner("ImpactArbiter — PagedAttention Block-Mapping Fuzzer v1.0", CYAN)

    # ── Print environment metadata ────────────────────────────────────────────
    section("Execution Environment")
    info(f"PyTorch version  : {torch.__version__}")
    info(f"SymPy  version   : {sympy.__version__}")
    info(f"Physical blocks  : {NUM_PHYSICAL_BLOCKS}")
    info(f"Block size       : {BLOCK_SIZE} tokens")
    info(f"KV-cache shape   : [{NUM_PHYSICAL_BLOCKS}, {BLOCK_SIZE}, {KV_CACHE_HEAD_STRIDE}]  (float32, requires_grad=True)")
    info(f"Dummy query dim  : {KV_CACHE_HEAD_STRIDE}")

    # ── Verify Oracle compiles ────────────────────────────────────────────────
    section("Oracle Compilation (SymPy → PyTorch lambdify)")
    test_lb, test_off = oracle_indices(token_idx=37)
    info(f"Oracle smoke-test  token_idx=37  →  logical_block={test_lb}, offset={test_off}")
    expected_lb  = 37 // BLOCK_SIZE
    expected_off = 37 % BLOCK_SIZE
    assert test_lb == expected_lb and test_off == expected_off, \
        f"Oracle self-check failed! Got ({test_lb},{test_off}), expected ({expected_lb},{expected_off})"
    ok("Oracle compiled and verified.")

    # ── Fuzzing loop ──────────────────────────────────────────────────────────
    banner("Fuzzing 20 Synthetic AI Agents Against the SymPy Oracle", YELLOW)

    passed = 0
    failed = 0
    results_log: list[dict] = []

    for agent_id, (agent_fn, label, is_correct) in enumerate(AGENT_REGISTRY, start=1):

        # Pick a random token index in valid range
        token_idx = random.randint(0, NUM_PHYSICAL_BLOCKS * BLOCK_SIZE - 1)

        print(f"\n{DIM}{'─' * 78}{RESET}")
        print(
            f"  {BOLD}Agent #{agent_id:02d}{RESET} "
            f"│ token_idx={token_idx:>3} "
            f"│ {YELLOW}{label}{RESET}"
        )

        # ── Oracle gradient ───────────────────────────────────────────────────
        oracle_lb, oracle_off = oracle_indices(token_idx)
        oracle_grad_tensor    = oracle_grad(token_idx)

        print(
            f"  {CYAN}Oracle{RESET}  →  logical_block={oracle_lb:>2}, offset={oracle_off:>2}  "
            f"│  ∇KV max={oracle_grad_tensor.abs().max().item():.6f}"
        )

        # ── Agent gradient ────────────────────────────────────────────────────
        try:
            agent_lb, agent_off = agent_fn(token_idx, BLOCK_SIZE)
            # Clamp indices to valid range (hallucinations can produce OOB values)
            agent_lb  = max(0, min(int(agent_lb),  NUM_PHYSICAL_BLOCKS - 1))
            agent_off = max(0, min(int(agent_off), BLOCK_SIZE - 1))

            agent_grad_tensor = run_proxy(agent_lb, agent_off)

            print(
                f"  {MAGENTA}Agent {RESET}  →  logical_block={agent_lb:>2}, offset={agent_off:>2}  "
                f"│  ∇KV max={agent_grad_tensor.abs().max().item():.6f}"
            )

            # ── Gradient comparison ───────────────────────────────────────────
            grad_match = torch.allclose(
                oracle_grad_tensor,
                agent_grad_tensor,
                rtol=1e-05,
                atol=1e-08,
            )

        except Exception as exc:
            warn(f"Agent #{agent_id} raised exception: {exc}")
            grad_match = False
            agent_grad_tensor = None

        # ── Verdict ───────────────────────────────────────────────────────────
        if grad_match:
            passed += 1
            pass_alert(agent_id)
        else:
            failed += 1
            gradient_divergence_alert(agent_id, label)

        results_log.append({
            "agent_id":    agent_id,
            "label":       label,
            "is_correct":  is_correct,
            "token_idx":   token_idx,
            "oracle_lb":   oracle_lb,
            "oracle_off":  oracle_off,
            "grad_match":  grad_match,
        })

    # ── Final scoreboard ──────────────────────────────────────────────────────
    banner("Fuzzing Complete — Final Scoreboard", CYAN)

    total = len(AGENT_REGISTRY)
    print(f"  {'Total agents evaluated':<35}: {total}")
    print(f"  {GREEN}{'✔ Gradient match (PASS)':<35}{RESET}: {passed}")
    print(f"  {RED}{'✘ Gradient divergence (FAIL)':<35}{RESET}: {failed}")
    print(f"  {'Pass rate':<35}: {100*passed/total:.1f}%")

    print(f"\n  {DIM}Per-agent breakdown:{RESET}")
    print(f"  {'ID':>4}  {'Pass':>5}  {'Oracle (lb,off)':>16}  Label")
    print(f"  {'─'*4}  {'─'*5}  {'─'*16}  {'─'*40}")
    for r in results_log:
        verdict_str  = f"{GREEN}PASS{RESET}" if r["grad_match"] else f"{RED}FAIL{RESET}"
        oracle_coords = f"({r['oracle_lb']:>2},{r['oracle_off']:>2})"
        print(
            f"  {r['agent_id']:>4}  {verdict_str}   {oracle_coords:>16}  "
            f"{r['label']}"
        )

    print(f"\n{BOLD}{CYAN}ImpactArbiter fuzzing session finished.{RESET}\n")


if __name__ == "__main__":
    main()
