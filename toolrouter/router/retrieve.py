"""Retrieval: query -> ranked ``ScoredTool`` candidates.

Owns the :class:`ScoredTool` and :class:`RouteResult` types (per
``ARCHITECTURE.md`` these are defined here and imported everywhere else).

Dense retrieval is the default. Hybrid retrieval fuses dense cosine scores with
BM25 lexical scores, min-max normalising **each list independently** first --
raw BM25 scores are unbounded and averaging them against cosine similarities
directly would let BM25 silently dominate the ranking.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..parser.manifest_parser import Tool

if TYPE_CHECKING:  # pragma: no cover
    from ..index.bm25 import BM25Index
    from ..index.embed import EmbeddingModel
    from ..index.vector_store import VectorStore
    from ..parser.tool_registry import ToolRegistry

__all__ = ["ScoredTool", "RouteResult", "Retriever", "DEFAULT_DENSE_WEIGHT"]

logger = logging.getLogger(__name__)

#: Weight on the dense score during hybrid fusion. 0.7/0.3 favours semantics
#: while letting exact lexical matches break near-ties -- see BENCHMARK results
#: for the measured effect.
DEFAULT_DENSE_WEIGHT = 0.7


@dataclass
class ScoredTool:
    """A retrieval candidate with its score and provenance."""

    tool: Tool
    score: float
    source: str  # "dense" | "bm25" | "hybrid"
    #: Per-signal component scores, populated for hybrid so ``explain`` can
    #: report *why* a tool ranked where it did rather than just the fused number.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool.name

    def to_dict(self) -> dict:
        return {
            "tool": self.tool.name,
            "server": self.tool.server,
            "score": round(float(self.score), 6),
            "source": self.source,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
        }


@dataclass
class RouteResult:
    """The full outcome of a routing call."""

    query: str
    tools: list[Tool]  # final selected tools, post-gate
    scored: list[ScoredTool]  # the gated candidate list, with scores
    explanation: list[dict]  # one dict per candidate, see explain.py
    #: Everything retrieved before the confidence gate ran -- kept so the gate's
    #: decision is auditable instead of opaque.
    candidates: list[ScoredTool] = field(default_factory=list)
    #: Gate decision metadata: mode, score gap, thresholds applied.
    gate: dict = field(default_factory=dict)
    #: Wall-clock retrieval latency in milliseconds.
    latency_ms: float = 0.0

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    @property
    def top_tool(self) -> Tool | None:
        return self.tools[0] if self.tools else None

    @property
    def is_confident(self) -> bool:
        """``True`` when the gate narrowed to a single confident tool."""
        return self.gate.get("mode") == "confident"

    @property
    def is_ambiguous(self) -> bool:
        """``True`` when the gate widened k because candidates were close."""
        return self.gate.get("mode") == "ambiguous"

    @property
    def no_confident_match(self) -> bool:
        """``True`` when every candidate fell below the absolute score floor."""
        return self.gate.get("mode") == "no_confident_match"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "tools": self.tool_names,
            "scored": [s.to_dict() for s in self.scored],
            "candidates": [c.to_dict() for c in self.candidates],
            "explanation": self.explanation,
            "gate": self.gate,
            "latency_ms": round(self.latency_ms, 3),
        }


def _min_max_normalize(scores: Sequence[float]) -> list[float]:
    """Min-max normalise into ``[0, 1]``.

    A flat list (all values equal) maps to all-``1.0`` rather than all-``0.0``:
    if every candidate is equally good lexically, that signal should not be
    zeroed out of the fusion.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo <= 1e-12:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    span = hi - lo
    return [(s - lo) / span for s in scores]


class Retriever:
    """Ranks tools for a query using dense and optionally lexical signals."""

    def __init__(
        self,
        registry: ToolRegistry,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        bm25: BM25Index | None = None,
        *,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError(f"dense_weight must be in [0, 1], got {dense_weight}.")
        self.registry = registry
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.dense_weight = float(dense_weight)

    # -- main entry point --------------------------------------------------- #
    def retrieve(
        self,
        query: str,
        k: int = 5,
        hybrid: bool = False,
        *,
        server: str | None = None,
    ) -> list[ScoredTool]:
        """Return up to ``k`` ranked candidates for ``query``.

        Parameters
        ----------
        query:
            Natural-language user request.
        k:
            Maximum candidates to return.
        hybrid:
            Fuse BM25 lexical scores with dense scores. Ignored (with a warning)
            when no BM25 index was configured.
        server:
            Optional post-filter restricting results to a single MCP server.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")
        if k is None or k <= 0:
            return []
        if not query.strip():
            return []

        total_tools = len(self.registry)
        if total_tools == 0:
            return []

        use_hybrid = bool(hybrid)
        if use_hybrid and self.bm25 is None:
            logger.warning("hybrid=True but no BM25 index configured; using dense only.")
            use_hybrid = False

        # Over-fetch so fusion and the gate see a real tail, not a pre-truncated
        # list -- fusion can promote a tool that dense ranked 8th.
        fetch_k = total_tools if use_hybrid else min(total_tools, max(int(k) * 4, 20))

        query_vector = self.embedder.embed_text(query)
        dense_hits = dict(self.vector_store.search(query_vector, k=fetch_k))

        if not use_hybrid:
            ranked = self._to_scored(dense_hits, source="dense")
        else:
            bm25_hits = dict(self.bm25.search(query, k=fetch_k))  # type: ignore[union-attr]
            ranked = self._fuse(dense_hits, bm25_hits)

        if server is not None:
            ranked = [c for c in ranked if c.tool.server == server]

        return ranked[: int(k)]

    # -- internals ---------------------------------------------------------- #
    def _to_scored(self, hits: dict[str, float], *, source: str) -> list[ScoredTool]:
        scored: list[ScoredTool] = []
        for tool_name, score in hits.items():
            tool = self.registry.by_name(tool_name)
            if tool is None:
                # Index and registry out of sync -- skip rather than crash the query.
                logger.warning("Index returned unknown tool %r; skipping.", tool_name)
                continue
            scored.append(
                ScoredTool(
                    tool=tool,
                    score=float(score),
                    source=source,
                    components={source: float(score)},
                )
            )
        scored.sort(key=lambda c: (-c.score, c.tool.name))
        return scored

    def _fuse(
        self, dense_hits: dict[str, float], bm25_hits: dict[str, float]
    ) -> list[ScoredTool]:
        """Weighted fusion of independently normalised dense and BM25 scores."""
        names = list(dict.fromkeys([*dense_hits.keys(), *bm25_hits.keys()]))
        if not names:
            return []

        # strict=True throughout: _min_max_normalize is order- and
        # length-preserving, so a mismatch here would mean a normalised score
        # got attached to the wrong tool name -- a silent ranking corruption.
        dense_norm = dict(
            zip(
                names,
                _min_max_normalize([dense_hits.get(n, 0.0) for n in names]),
                strict=True,
            )
        )
        # Tools absent from the BM25 result list genuinely scored zero lexically;
        # normalise over the full name set so that zero stays meaningful.
        bm25_norm = dict(
            zip(
                names,
                _min_max_normalize([bm25_hits.get(n, 0.0) for n in names]),
                strict=True,
            )
        )

        w = self.dense_weight
        scored: list[ScoredTool] = []
        for name in names:
            tool = self.registry.by_name(name)
            if tool is None:
                logger.warning("Index returned unknown tool %r; skipping.", name)
                continue
            dense_score = float(dense_norm.get(name, 0.0))
            lexical_score = float(bm25_norm.get(name, 0.0))
            scored.append(
                ScoredTool(
                    tool=tool,
                    score=w * dense_score + (1.0 - w) * lexical_score,
                    source="hybrid",
                    components={
                        "dense": dense_score,
                        "bm25": lexical_score,
                        "dense_raw": float(dense_hits.get(name, 0.0)),
                        "bm25_raw": float(bm25_hits.get(name, 0.0)),
                    },
                )
            )
        scored.sort(key=lambda c: (-c.score, c.tool.name))
        return scored

    def __repr__(self) -> str:
        return (
            f"Retriever(tools={len(self.registry)}, hybrid_available={self.bm25 is not None}, "
            f"dense_weight={self.dense_weight})"
        )
