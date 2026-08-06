"""Explainability: honest, template-based reasons per candidate.

Constraint from ``ARCHITECTURE.md``/``PROMPTS.md``: reasons must cite only
signals the retrieval process *actually used*. No "previously successful", no
"learned from feedback" -- there is no historical-success tracking in this
version, and fabricating such a reason would make the explanation output
actively misleading.

What we can honestly report: the score, which query terms overlapped the tool's
name / description / parameter names, the owning server, whether the score came
from dense, lexical, or fused signals, and the candidate's rank.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..index.bm25 import tokenize
from .retrieve import ScoredTool

if TYPE_CHECKING:  # pragma: no cover
    from ..parser.tool_registry import ToolRegistry

__all__ = ["explain_candidates", "STOPWORDS"]

#: Function words carry no disambiguating signal; excluding them keeps the
#: "matched terms" list meaningful rather than reporting "the, a, for".
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do",
        "for", "from", "get", "give", "has", "have", "how", "i", "in", "is", "it",
        "its", "me", "my", "of", "on", "or", "please", "the", "then", "there",
        "this", "to", "want", "was", "we", "what", "when", "where", "which",
        "will", "with", "would", "you", "your",
    }
)


def _content_terms(text: str) -> list[str]:
    """Tokenise and drop stopwords / 1-char noise, preserving order."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(text):
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _overlap(query_terms: Iterable[str], target: str) -> list[str]:
    target_tokens = set(tokenize(target))
    return [term for term in query_terms if term in target_tokens]


def _score_band(score: float) -> str:
    """Coarse qualitative label for a score. Deliberately vague -- the exact
    number is reported alongside it, so this is only a reading aid."""
    if score >= 0.85:
        return "very high"
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "moderate"
    return "low"


def _source_phrase(candidate: ScoredTool) -> str:
    if candidate.source == "hybrid":
        dense = candidate.components.get("dense")
        bm25 = candidate.components.get("bm25")
        if dense is not None and bm25 is not None:
            return (
                f"fused dense+lexical score (dense {dense:.2f}, BM25 {bm25:.2f}, "
                "both min-max normalised)"
            )
        return "fused dense+lexical score"
    if candidate.source == "bm25":
        return "lexical BM25 score"
    return "dense embedding cosine similarity"


def explain_candidates(
    query: str,
    scored: list[ScoredTool],
    registry: ToolRegistry | None = None,
    *,
    gate: dict | None = None,
) -> list[dict]:
    """Build one explanation dict per candidate.

    Each dict contains::

        {
          "rank": 1,
          "tool": "book_table",
          "server": "dineout",
          "score": 0.91,
          "score_band": "very high",
          "source": "dense",
          "matched_terms": {"name": ["book", "table"], "description": ["book"],
                            "parameters": ["party", "size"]},
          "reason": "Ranked #1 with a very high dense embedding cosine similarity "
                    "of 0.91; query terms 'book', 'table' appear in the tool name."
        }

    ``gate`` is the optional :meth:`GateDecision.to_dict` output; when supplied,
    its ``reason`` is attached to every row as ``gate_reason`` so a single
    explanation payload tells the whole story.
    """
    query_terms = _content_terms(query or "")
    explanations: list[dict] = []

    for rank, candidate in enumerate(scored, start=1):
        tool = candidate.tool
        name_hits = _overlap(query_terms, tool.name)
        desc_hits = _overlap(query_terms, tool.description)
        param_hits = _overlap(query_terms, " ".join(tool.parameter_names))
        server_hits = _overlap(query_terms, tool.server)

        score = float(candidate.score)
        band = _score_band(score)

        clauses: list[str] = [
            f"Ranked #{rank} with a {band} {_source_phrase(candidate)} of {score:.3f}"
        ]
        if name_hits:
            clauses.append(
                f"query term{'s' if len(name_hits) > 1 else ''} "
                + ", ".join(f"'{t}'" for t in name_hits)
                + " appear in the tool name"
            )
        if desc_hits:
            clauses.append(
                "description mentions " + ", ".join(f"'{t}'" for t in desc_hits)
            )
        if param_hits:
            clauses.append(
                "parameter names include " + ", ".join(f"'{t}'" for t in param_hits)
            )
        if server_hits:
            clauses.append(f"query references the '{tool.server}' server directly")
        if not (name_hits or desc_hits or param_hits or server_hits):
            clauses.append(
                "no literal term overlap -- selected on semantic similarity alone"
            )

        explanations.append(
            {
                "rank": rank,
                "tool": tool.name,
                "server": tool.server,
                "score": round(score, 6),
                "score_band": band,
                "source": candidate.source,
                "components": {
                    k: round(float(v), 6) for k, v in candidate.components.items()
                },
                "matched_terms": {
                    "name": name_hits,
                    "description": desc_hits,
                    "parameters": param_hits,
                    "server": server_hits,
                },
                "reason": "; ".join(clauses) + ".",
                **({"gate_reason": gate.get("reason", "")} if gate else {}),
            }
        )

    if not explanations:
        row = {
            "rank": 0,
            "tool": None,
            "server": None,
            "score": 0.0,
            "score_band": "none",
            "source": "none",
            "components": {},
            "matched_terms": {"name": [], "description": [], "parameters": [], "server": []},
            "reason": (
                "No tool was selected for this query. Either retrieval returned "
                "nothing, or every candidate scored below the confidence gate's "
                "absolute floor."
            ),
        }
        if gate:
            row["gate_reason"] = gate.get("reason", "")
        explanations.append(row)

    return explanations
