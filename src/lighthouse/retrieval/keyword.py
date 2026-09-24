"""Identifier-preserving tokenisation.

From *Retrieval-Augmented Generation (RAG) in Production*, Chapter 12.
"""
from __future__ import annotations

import re

TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
IDENTIFIER = re.compile(r"\b[A-Z]{2,4}-\d{3,5}(?:-[A-Z])?\b")


def tokenize(text: str) -> list[str]:
    """Keep hyphenated alphanumeric runs intact so END-4471-B
    survives as one rare token (Chapter 12, section 12.2)."""
    return [t.lower() for t in TOKEN.findall(text)]
