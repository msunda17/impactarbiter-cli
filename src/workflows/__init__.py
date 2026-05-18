"""Reference workflows for Application Architects.

Each module here is a self-contained, runnable reference architecture that
demonstrates how to wrap real agentic-serving primitives with the
``@verify`` decorator from ``src.core``.
"""

from .agentic_kv_scheduler import main as run_agentic_kv_scheduler

__all__ = ["run_agentic_kv_scheduler"]
