"""
recency.py — date-aware ranking helpers shared by the retriever and reranker
agents.

The IDiOM-RAG corpus spans roughly 1989-2025 of Office Memoranda where many
topics are addressed repeatedly across years (stipend revisions, allowance
schedules, leadership appointments). When a query has no explicit time
scope, the system should prefer the most recent OM. These helpers compute
a multiplicative recency boost that is applied alongside (not instead of)
the underlying relevance score.

Design choices
--------------
- Boost is LINEAR in (year - oldest) / (newest - oldest), not exponential,
  so an early-2010s OM is not crushed against a 2025 OM when they're both
  relevant — it's a tiebreaker, not a kill switch.
- Boost is applied AFTER the underlying retriever / cross-encoder score,
  as a multiplier (1 + lambda * recency). lambda is small (~0.15 at the
  retriever, ~0.30 at the reranker) so highly relevant old OMs still survive.
- Documents with missing / UNKNOWN issue_date receive recency_weight=0.5
  (neutral): not penalised, but not boosted either. They sink below recent
  documents and rise above clearly-stale ones.
- Supersession demotion is multiplicative and composes naturally with the
  recency boost (apply_recency then apply_supersession, or vice versa,
  yields the same final score).
"""

from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document

import config


_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})")


def _extract_year(doc: Document) -> int | None:
    """Return the OM's issue year, or None if unparseable / UNKNOWN.

    Reads `doc.metadata['issue_date']` which the ingest pipeline writes as
    YYYY-MM-DD (or the literal 'UNKNOWN')."""
    if not getattr(doc, "metadata", None):
        return None
    raw = str(doc.metadata.get("issue_date", "")).strip()
    if not raw or raw.upper() == "UNKNOWN":
        return None
    m = _ISO_DATE_RE.match(raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def recency_weight(doc: Document) -> float:
    """Return a value in [0, 1] representing how recent this OM is.

    1.0 = published in CORPUS_NEWEST_YEAR.
    0.0 = published in CORPUS_OLDEST_YEAR.
    0.5 = issue_date is UNKNOWN (neutral; neither boosted nor penalised).
    Years outside the configured bounds are clamped."""
    year = _extract_year(doc)
    if year is None:
        return 0.5
    oldest = config.CORPUS_OLDEST_YEAR
    newest = config.CORPUS_NEWEST_YEAR
    if newest <= oldest:                          # mis-configured; degenerate to neutral
        return 0.5
    raw = (year - oldest) / (newest - oldest)
    return max(0.0, min(1.0, raw))


def is_superseded(doc: Document) -> bool:
    """Did the reconcile_supersession pass flip this OM's flag?"""
    if not getattr(doc, "metadata", None):
        return False
    return str(doc.metadata.get("superseded", "0")).strip() == "1"


def apply_recency_boost(docs: Iterable[Document],
                        base_scores: Iterable[float],
                        lam: float) -> list[tuple[Document, float]]:
    """Multiply each base score by (1 + lam * recency_weight(doc)) and also
    apply the supersession penalty when the flag is set.

    Returns (doc, adjusted_score) pairs in the SAME order as input — sorting
    is the caller's responsibility, because retriever agents and the reranker
    agent want different post-processing (e.g. the reranker also writes the
    final score back into metadata for downstream consumers).
    """
    out: list[tuple[Document, float]] = []
    for doc, base in zip(docs, base_scores):
        s = float(base) * (1.0 + lam * recency_weight(doc))
        if is_superseded(doc):
            s *= config.SUPERSEDED_PENALTY
        out.append((doc, s))
    return out


def rank_by_recency_boost(docs: list[Document],
                          base_scores: list[float],
                          lam: float,
                          top_k: int) -> list[Document]:
    """Convenience wrapper used by the retriever agents.

    Applies the recency boost + supersession penalty, sorts descending, and
    returns the top_k documents. Also stashes the recency_weight and the
    final boosted score in metadata so downstream agents (reranker, critic)
    can introspect them.
    """
    scored = apply_recency_boost(docs, base_scores, lam)
    scored.sort(key=lambda ds: ds[1], reverse=True)
    top = scored[:top_k]
    for doc, score in top:
        if not getattr(doc, "metadata", None):
            doc.metadata = {}
        doc.metadata["recency_weight"] = round(recency_weight(doc), 3)
        doc.metadata["retriever_boosted_score"] = round(score, 4)
    return [doc for doc, _ in top]


def proxy_relevance_score(rank: int, n: int) -> float:
    """When a retriever does not return per-document scores (e.g. BM25Retriever
    just returns a ranked list), fabricate a monotonically decreasing relevance
    proxy so the recency boost has something to multiply against.

    Uses a linear decay from 1.0 (rank 0) down to 1/n (last rank) so the
    boost can meaningfully reorder ties without inventing magnitude where
    none exists.
    """
    if n <= 0:
        return 0.0
    return 1.0 - (rank / max(n, 1)) * (1.0 - 1.0 / n)
