"""semantic_memory: extractive recall over a personal message archive.

Public API::

    from semantic_memory import build_index, select, load_messages

    msgs = load_messages("data/messages.jsonl")
    X, idx, idf = build_index(msgs)
    blocks, stats = select("loocv leave-one-out", msgs, X, idx, idf, {})
"""

from .features import build_index, grams, query_vector
from .recall import (select, score, load_messages, load_state, save_state,
                     print_result, fmt_time)
from .session import Session

__version__ = "0.1.0"
__all__ = [
    "build_index", "grams", "query_vector",
    "select", "score", "load_messages", "load_state", "save_state",
    "print_result", "fmt_time", "Session",
]
