"""Format-aware prose audit records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Passage:
    """One authored prose passage with a stable source location."""

    path: str
    line: int
    kind: str
    owner: str
    text: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """One named pattern candidate requiring contextual adjudication."""

    passage: Passage
    rule_id: str
    rule_name: str
    source: str
    confidence: str
    evidence: str


__all__ = ["Candidate", "Passage"]
