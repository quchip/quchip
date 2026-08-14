"""Named prose-pattern candidates derived from the two reviewed style guides."""

from __future__ import annotations

from dataclasses import dataclass
import re

from . import Candidate, Passage


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    name: str
    source: str
    confidence: str
    pattern: re.Pattern[str]


def _rule(
    rule_id: str,
    name: str,
    source: str,
    confidence: str,
    pattern: str,
) -> _Rule:
    return _Rule(rule_id, name, source, confidence, re.compile(pattern, re.IGNORECASE | re.MULTILINE))


_HIGH_CONFIDENCE_RULES = (
    _rule(
        "throat-clearing",
        "Throat-clearing opener",
        "no-ai-slop + stop-slop",
        "high",
        r"(?:^|[.!?]\s+)(?:here(?:'|’)s (?:the thing|what|why)|let me be clear|the uncomfortable truth is|"
        r"i(?:'|’)ll be honest|it turns out(?: that)?)\b",
    ),
    _rule(
        "binary-contrast",
        "Binary contrast",
        "no-ai-slop + stop-slop",
        "high",
        r"\b(?:the (?:question|answer|problem) is(?:n(?:'|’)t| not)|it is(?:n(?:'|’)t| not)|"
        r"not because)\b[^.!?]{0,140}[.!?,]\s*(?:it is|it(?:'|’)s|because)\b|"
        r"\bnot just\b[^.!?]{0,100}\bbut(?: also)?\b",
    ),
    _rule(
        "faux-insight",
        "Faux-insight setup",
        "no-ai-slop",
        "high",
        r"\b(?:what (?:most people|nobody) (?:get wrong|tell you)|the part (?:everyone|most people) (?:misses|skip))\b",
    ),
    _rule(
        "importance-puffery",
        "Importance puffery",
        "no-ai-slop",
        "high",
        r"\b(?:marks? a pivotal moment|stands? as a testament|plays? a vital role|"
        r"solidif(?:y|ies) its position|underscores? its significance|paramount|transformative)\b",
    ),
    _rule(
        "weasel-attribution",
        "Weasel attribution",
        "no-ai-slop",
        "high",
        r"\b(?:experts agree|studies show|industry reports suggest|many argue|widely regarded as)\b",
    ),
    _rule(
        "interpretive-metadiscourse",
        "Interpretive metadiscourse",
        "no-ai-slop",
        "high",
        r"\b(?:this distinction matters|that last part matters|the key point is|as you can see|"
        r"this matters because|here(?:'|’)s why that matters)\b",
    ),
    _rule(
        "vague-declaration",
        "Vague declaration",
        "stop-slop",
        "high",
        r"\b(?:the reasons are structural|the implications are significant|this is the deepest problem|"
        r"the stakes are high|the consequences are real)\b",
    ),
    _rule(
        "business-jargon",
        "Generic business jargon",
        "no-ai-slop + stop-slop",
        "high",
        r"\b(?:delve|foster|leverage|utilize|facilitate|empower|streamline|robust|cutting-edge|"
        r"paradigm shift|game[ -]changer|tapestry|realm|beacon|multifaceted|meticulous|intricate|"
        r"elevate|embark|supercharge|harness|ever-evolving|lean into|double down|deep dive)\b",
    ),
    _rule(
        "dramatic-fragmentation",
        "Dramatic fragmentation",
        "no-ai-slop + stop-slop",
        "high",
        r"(?:\b[A-Z][A-Za-z-]+\.\s+){2,}[A-Z][A-Za-z-]+\.\s+that(?:'|’)s it\b|"
        r"\bthat(?:'|’)s it\.\s+that(?:'|’)s the\b|"
        r"(?:^|[.!?]\s+)and\s+[^.!?]{1,50}[.!?]\s+and\s+[^.!?]{1,50}[.!?]",
    ),
    _rule(
        "colon-reveal",
        "Colon reveal",
        "no-ai-slop",
        "high",
        r"\b(?:the best part|the key point|the detail that makes it work|the answer|the result):\s+[a-z]",
    ),
    _rule(
        "recap-ending",
        "Summary-recap ending",
        "no-ai-slop",
        "high",
        r"(?:^|\n\s*|[.!?]\s+)(?:in conclusion|ultimately|overall),",
    ),
)

_REVIEW_SIGNAL_RULES = (
    _rule(
        "passive-voice",
        "Possible passive voice",
        "stop-slop",
        "contextual",
        r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b",
    ),
    _rule(
        "adverb",
        "Possible empty adverb",
        "stop-slop",
        "contextual",
        r"\b(?:really|just|literally|genuinely|honestly|simply|actually|deeply|truly|fundamentally|"
        r"inherently|inevitably|interestingly|importantly|crucially|carefully)\b",
    ),
    _rule("em-dash", "Em-dash usage", "stop-slop", "contextual", r"—"),
    _rule(
        "wh-opener",
        "Wh-word sentence opener",
        "stop-slop",
        "contextual",
        r"(?:^|[.!?]\s+)(?:what|when|where|which|who|why|how)\b",
    ),
    _rule(
        "three-item-list",
        "Three-item list",
        "stop-slop",
        "contextual",
        r"\b[^,.;\n]{1,50},\s+[^,.;\n]{1,50},\s+(?:and|or)\s+[^,.;\n]{1,50}[.!?]?",
    ),
    _rule(
        "inanimate-agency",
        "Possible inanimate agency",
        "stop-slop",
        "contextual",
        r"\b(?:the (?:decision|data|market|conversation|culture)|a (?:complaint|bet))\s+"
        r"(?:emerges?|tells?|rewards?|moves?|shifts?|becomes?|lives?|dies?)\b",
    ),
)


def _evidence_for(text: str, match: re.Match[str]) -> str:
    start = max(text.rfind("\n", 0, match.start()), text.rfind(". ", 0, match.start()))
    start = 0 if start < 0 else start + (2 if text[start : start + 2] == ". " else 1)
    endings = [index for token in (". ", "? ", "! ", "\n") if (index := text.find(token, match.end())) >= 0]
    end = min(endings) + 1 if endings else len(text)
    return " ".join(text[start:end].split())


def find_candidates(passage: Passage) -> list[Candidate]:
    """Return named candidates without adjudicating or rewriting the passage."""
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for rule in (*_HIGH_CONFIDENCE_RULES, *_REVIEW_SIGNAL_RULES):
        for match in rule.pattern.finditer(passage.text):
            evidence = _evidence_for(passage.text, match)
            key = (rule.rule_id, evidence.casefold())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                Candidate(
                    passage=passage,
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    source=rule.source,
                    confidence=rule.confidence,
                    evidence=evidence,
                )
            )
    return sorted(candidates, key=lambda item: (item.passage.line, item.rule_id, item.evidence.casefold()))
