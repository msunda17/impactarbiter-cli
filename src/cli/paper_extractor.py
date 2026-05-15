"""Paper extraction using pypdf for two-stage RAG pipeline.

Downloads ArXiv PDFs and extracts text chunks containing "KV cache" and "routing"
for the Distill Agent to summarize.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

ARXIV_URLS = {
    "radix": "https://arxiv.org/pdf/2312.07104.pdf",  # Zheng et al. — SGLang / RadixAttention (DEFAULT)
    "vllm": "https://arxiv.org/pdf/2309.06180.pdf",   # Kwon et al. — vLLM PagedAttention (LEGACY)
}

ANCHOR_KEYWORDS = {
    "radix": ["RadixAttention", "radix tree", "prefix", "shared prefix", "prefix cache"],
    "vllm": ["PagedAttention", "logical block", "block table", "KV cache"],
}

# Regex that strips lines containing an explicit modulo/floor-division formula —
# we don't want to hand the LLM the answer.
_FORMULA_LINE = re.compile(
    r"(?://\s*block_size|%\s*block_size|floor\(|⌊|\bmod\b\s*block_size)",
    re.IGNORECASE,
)


def _load_paper_text(url: str) -> str:
    """Download and extract text from PDF using pypdf."""
    try:
        import requests
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf and requests are required for paper extraction. "
            "Install with: pip install pypdf requests"
        ) from e

    # Download PDF
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Extract text
    reader = PdfReader(response.content)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text


def _find_anchor_window(
    text: str,
    anchors: List[str],
    *,
    max_chars: int,
) -> Tuple[int, int]:
    """Return (start, end) char offsets centered on the first anchor hit."""
    lower = text.lower()
    for anchor in anchors:
        idx = lower.find(anchor.lower())
        if idx >= 0:
            start = max(0, idx - max_chars // 4)
            end = min(len(text), start + max_chars)
            return start, end
    # Fallback: top of paper.
    return 0, min(len(text), max_chars)


def _strip_formula_lines(snippet: str) -> str:
    """Remove lines that contain explicit block_size arithmetic."""
    kept: List[str] = []
    redactions = 0
    for line in snippet.splitlines():
        if _FORMULA_LINE.search(line):
            redactions += 1
            continue
        kept.append(line)
    out = "\n".join(kept)
    if redactions:
        out += f"\n\n[note: {redactions} line(s) containing explicit block-size arithmetic were redacted from this excerpt]"
    return out


def get_paper_excerpt(oracle_kind: str, *, max_chars: int = 6000) -> str:
    """Return a real, formula-redacted excerpt for the given oracle.

    Falls back to a short prose summary if the PDF cannot be fetched (e.g. no
    network in CI). The summary intentionally omits the numerical formula.
    """
    fallbacks = {
        "radix": (
            "RadixAttention (SGLang, Zheng et al., 2023) accelerates serving "
            "by sharing KV-cache prefixes across requests via a radix tree "
            "keyed on the token sequence. When a new request reuses a prefix, "
            "its first generated token must be placed immediately after the "
            "shared prefix in the existing physical block — even when the "
            "prefix length is not a multiple of the block size, in which case "
            "the executor continues writing into the partially-filled block "
            "the planner left behind. Only once that partial block fills does "
            "the executor advance to the next block. Misrouting at this "
            "ragged boundary is silent: shapes match, returns look plausible, "
            "but the gradient through the KV cache deviates from the "
            "ground-truth mapping."
        ),
        "vllm": (
            "PagedAttention (Kwon et al., 2023) treats the KV cache as a "
            "collection of fixed-size logical blocks that are mapped to "
            "physical blocks via a per-sequence block table. Each token in the "
            "logical sequence is assigned to a logical block according to its "
            "position, and within that block it occupies a slot determined by "
            "its position relative to the block's start. The block table is "
            "consulted at attention time to dereference the correct physical "
            "page. The paper emphasises that the mapping must be exact — "
            "off-by-one routing of even a single token corrupts attention for "
            "the rest of the sequence."
        ),
    }
    if oracle_kind not in ARXIV_URLS:
        raise ValueError(f"unknown oracle kind: {oracle_kind!r}")

    if os.environ.get("IMPACTARBITER_OFFLINE", "").lower() in ("1", "true", "yes"):
        return fallbacks[oracle_kind]

    try:
        full = _load_paper_text(ARXIV_URLS[oracle_kind])
    except Exception:  # noqa: BLE001 — network/pypdf unavailable
        return fallbacks[oracle_kind]

    start, end = _find_anchor_window(
        full, ANCHOR_KEYWORDS[oracle_kind], max_chars=max_chars
    )
    excerpt = full[start:end]
    excerpt = _strip_formula_lines(excerpt)
    if not excerpt.strip():
        return fallbacks[oracle_kind]
    return excerpt


__all__ = [
    "ARXIV_URLS",
    "get_paper_excerpt",
]
