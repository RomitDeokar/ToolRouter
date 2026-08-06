"""
toolrouter
==========

A semantic retrieval layer for MCP agents: retrieve only the most relevant
tools before the LLM reasons, instead of stuffing every tool schema into
every prompt.

Quickstart
----------
    from toolrouter import ToolRouter

    router = ToolRouter.from_manifest("examples/swiggy_manifest.json")
    result = router.route("Book an Italian restaurant for tonight")

    print(result.tool_names)   # routed tools, post confidence gate
    print(result.explanation)  # why each one was chosen
    print(router.build_prompt(result))  # what the LLM actually sees
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from typing import Any

from .index.bm25 import BM25Index
from .index.embed import EmbeddingModel
from .index.vector_store import VectorStore
from .parser.manifest_parser import (
    ManifestError,
    Tool,
    parse_manifest,
    parse_manifest_dict,
)
from .parser.tool_registry import ToolRegistry
from .router.confidence_gate import (
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_SCORE_FLOOR,
    GateDecision,
    GateMode,
    apply_confidence_gate,
    confidence_gate,
)
from .router.explain import explain_candidates
from .router.prompt_builder import (
    build_no_match_prompt,
    build_tool_prompt,
    estimate_tokens,
    tokens_for_tools,
)
from .router.retrieve import Retriever, RouteResult, ScoredTool

__all__ = [
    "ToolRouter",
    "Tool",
    "ToolRegistry",
    "ManifestError",
    "EmbeddingModel",
    "VectorStore",
    "BM25Index",
    "Retriever",
    "RouteResult",
    "ScoredTool",
    "GateDecision",
    "GateMode",
    "confidence_gate",
    "apply_confidence_gate",
    "explain_candidates",
    "build_tool_prompt",
    "build_no_match_prompt",
    "estimate_tokens",
    "tokens_for_tools",
    "parse_manifest",
    "parse_manifest_dict",
    # Lazily loaded -- see __getattr__ below.
    "RoutedAgent",
    "AgentRun",
    "to_openai_tools",
    "__version__",
]

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

#: Names resolved lazily from :mod:`toolrouter.agent`. The agent module is the
#: only part of the package that may reach for an optional LLM SDK, so importing
#: ``toolrouter`` must not pull it in -- users who only want retrieval should
#: never pay for it.
_LAZY_EXPORTS = {
    "RoutedAgent": "agent",
    "AgentRun": "agent",
    "AgentStep": "agent",
    "HeuristicClient": "agent",
    "OpenAIClient": "agent",
    "EchoToolExecutor": "agent",
    "to_openai_tools": "agent",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for the agent layer."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


class ToolRouter:
    """High-level facade -- the only class most users need to touch.

    Wires together: manifest parsing -> registry -> embeddings -> vector index
    -> (optional) BM25 -> retriever -> confidence gate -> explanations ->
    prompt block.

    Parameters
    ----------
    registry:
        Parsed tools. Use :meth:`from_manifest` / :meth:`from_tools` instead of
        constructing this by hand in normal use.
    use_hybrid:
        Build a BM25 index alongside the dense index and fuse both at query
        time by default.
    embedder / vector_store:
        Injectable for testing and for sharing one loaded model across several
        routers (loading an embedding model is the expensive part).
    gap_threshold / score_floor / min_k / max_k:
        Confidence-gate defaults for this router instance; every one can be
        overridden per :meth:`route` call.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        use_hybrid: bool = False,
        embedder: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        dense_weight: float = 0.7,
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
        score_floor: float = DEFAULT_SCORE_FLOOR,
        min_k: int = 1,
        max_k: int = 5,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError(
                f"ToolRouter requires a ToolRegistry, got {type(registry).__name__}."
            )
        self.registry = registry
        self.use_hybrid = bool(use_hybrid)
        self.gap_threshold = gap_threshold
        self.score_floor = score_floor
        self.min_k = min_k
        self.max_k = max_k

        # Identity checks, not truthiness: an empty VectorStore is falsy
        # (``__len__`` == 0), so ``vector_store or VectorStore(...)`` would
        # silently discard an injected-but-empty store -- including one built
        # with a mismatched dim, which is exactly the error the check below
        # exists to catch.
        self.embedder = EmbeddingModel() if embedder is None else embedder
        self.vector_store = (
            VectorStore(dim=self.embedder.dim) if vector_store is None else vector_store
        )
        if self.vector_store.dim != self.embedder.dim:
            raise ValueError(
                f"Vector store dim ({self.vector_store.dim}) does not match embedder "
                f"dim ({self.embedder.dim})."
            )
        self.bm25: BM25Index | None = BM25Index() if self.use_hybrid else None

        self.index_tools()
        self.retriever = Retriever(
            registry=self.registry,
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25=self.bm25,
            dense_weight=dense_weight,
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_manifest(
        cls, path_or_url: str, *, use_hybrid: bool = False, **kwargs: Any
    ) -> ToolRouter:
        """Build a router from one manifest path/URL."""
        tools = parse_manifest(path_or_url)
        return cls(ToolRegistry(tools), use_hybrid=use_hybrid, **kwargs)

    @classmethod
    def from_manifests(
        cls, paths_or_urls: Sequence[str], *, use_hybrid: bool = False, **kwargs: Any
    ) -> ToolRouter:
        """Build a router across several MCP servers at once.

        This is the case the project is really aimed at: an agent wired to
        Swiggy *and* GitHub *and* Slack simultaneously, where the combined tool
        count makes stuffing every schema into context untenable.
        """
        tools: list[Tool] = []
        for location in paths_or_urls:
            tools.extend(parse_manifest(location))
        names = [t.name for t in tools]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ManifestError(
                f"Tool names collide across manifests: {sorted(duplicates)}. "
                "Namespace them before combining."
            )
        return cls(ToolRegistry(tools), use_hybrid=use_hybrid, **kwargs)

    @classmethod
    def from_tools(
        cls, tools: Iterable[Tool], *, use_hybrid: bool = False, **kwargs: Any
    ) -> ToolRouter:
        """Build a router from already-constructed :class:`Tool` objects."""
        return cls(ToolRegistry(list(tools)), use_hybrid=use_hybrid, **kwargs)

    # -- API.md compatibility ------------------------------------------- #
    def load_manifest(self, path_or_url: str) -> list[Tool]:
        """Load additional tools from a manifest and re-index in place.

        Present to satisfy the ``API.md`` surface (``load_manifest`` /
        ``index_tools`` / ``retrieve`` / ``explain`` / ``benchmark``).
        """
        new_tools = parse_manifest(path_or_url)
        existing = {t.name for t in self.registry.tools}
        collisions = sorted(t.name for t in new_tools if t.name in existing)
        if collisions:
            raise ManifestError(
                f"Manifest {path_or_url!r} re-declares existing tools: {collisions}."
            )
        self.registry = ToolRegistry([*self.registry.tools, *new_tools])
        self.vector_store.reset()
        self.index_tools()
        self.retriever.registry = self.registry
        return self.registry.tools

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_tools(self) -> None:
        """(Re)build the dense index -- and the BM25 index when hybrid is on."""
        ids, texts = self.registry.embedding_texts()
        if not ids:
            logger.warning("index_tools(): registry is empty; nothing indexed.")
            return
        vectors = self.embedder.embed_batch(texts)
        self.vector_store.add(ids, vectors)
        if self.bm25 is not None:
            self.bm25.build(ids, texts)
        logger.info(
            "Indexed %d tools across %d server(s) [embedder=%s, store=%s]",
            len(ids),
            len(self.registry.servers),
            self.embedder.backend,
            self.vector_store.backend,
        )

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        hybrid: bool | None = None,
        server: str | None = None,
    ) -> list[ScoredTool]:
        """Ranked candidates for ``query``, *without* applying the gate."""
        use_hybrid = self.use_hybrid if hybrid is None else hybrid
        return self.retriever.retrieve(query, k=k, hybrid=use_hybrid, server=server)

    def route(
        self,
        query: str,
        k: int | None = None,
        *,
        hybrid: bool | None = None,
        server: str | None = None,
        use_gate: bool = True,
        min_k: int | None = None,
        max_k: int | None = None,
        gap_threshold: float | None = None,
        score_floor: float | None = None,
    ) -> RouteResult:
        """Full pipeline: retrieve -> gate -> explain.

        Parameters
        ----------
        k:
            Candidate pool size. Defaults to ``max_k`` so the gate always has
            enough candidates to widen into.
        use_gate:
            Set ``False`` for plain fixed-top-k retrieval (this is exactly how
            the ``dense`` and ``hybrid`` benchmark baselines are run).
        """
        resolved_min_k = self.min_k if min_k is None else min_k
        resolved_max_k = self.max_k if max_k is None else max_k
        pool = resolved_max_k if k is None else k

        start = time.perf_counter()
        candidates = self.retrieve(query, k=pool, hybrid=hybrid, server=server)

        if use_gate:
            decision = apply_confidence_gate(
                candidates,
                min_k=resolved_min_k,
                max_k=resolved_max_k,
                gap_threshold=self.gap_threshold if gap_threshold is None else gap_threshold,
                score_floor=self.score_floor if score_floor is None else score_floor,
            )
            selected = decision.selected
            gate_info = decision.to_dict()
        else:
            selected = candidates[:pool]
            gate_info = {
                "mode": "disabled",
                "selected_k": len(selected),
                "reason": f"Confidence gate disabled; returning fixed top-{pool}.",
            }
        latency_ms = (time.perf_counter() - start) * 1000.0

        return RouteResult(
            query=query,
            tools=[c.tool for c in selected],
            scored=selected,
            explanation=explain_candidates(
                query, selected, registry=self.registry, gate=gate_info
            ),
            candidates=candidates,
            gate=gate_info,
            latency_ms=latency_ms,
        )

    def explain(self, query: str, **kwargs: Any) -> list[dict]:
        """Convenience wrapper returning just the explanation payload."""
        return self.route(query, **kwargs).explanation

    def build_prompt(
        self,
        result_or_query: RouteResult | str,
        *,
        style: str = "compact",
        include_examples: bool = False,
        header: bool = True,
        **route_kwargs: Any,
    ) -> str:
        """Render the routed tool subset as an LLM-ready prompt block.

        Pass ``header=False`` when embedding the block inside a larger system
        prompt that already provides its own framing (see
        :class:`toolrouter.agent.RoutedAgent`).
        """
        result = (
            self.route(result_or_query, **route_kwargs)
            if isinstance(result_or_query, str)
            else result_or_query
        )
        if not result.tools:
            return build_no_match_prompt()
        return build_tool_prompt(
            result.tools,
            style=style,
            include_examples=include_examples,
            header=header,
        )

    def all_tools_prompt(self, *, style: str = "json") -> str:
        """The unrouted baseline prompt: every tool in the registry."""
        return build_tool_prompt(self.registry.tools, style=style)

    # ------------------------------------------------------------------ #
    # Benchmark
    # ------------------------------------------------------------------ #
    def benchmark(
        self,
        dataset_path: str | None = None,
        *,
        baselines: Sequence[str] | None = None,
        output_dir: str = "bench_results",
    ) -> dict:
        """Run CommerceBench against this router.

        Imported lazily so the benchmark's dependencies never load for users who
        only want routing.
        """
        from .bench.evaluate import evaluate_all

        return evaluate_all(
            dataset_path=dataset_path,
            router=self,
            baselines=baselines,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def tools(self) -> list[Tool]:
        return self.registry.tools

    def stats(self) -> dict:
        """Configuration snapshot -- written into every benchmark result file."""
        return {
            "tools": len(self.registry),
            "servers": self.registry.servers,
            "embedder": self.embedder.backend,
            "embedder_dim": self.embedder.dim,
            "embedder_is_fallback": self.embedder.is_fallback,
            "vector_store": self.vector_store.backend,
            "bm25": self.bm25.backend if self.bm25 else None,
            "gap_threshold": self.gap_threshold,
            "score_floor": self.score_floor,
            "min_k": self.min_k,
            "max_k": self.max_k,
            "version": __version__,
        }

    def __repr__(self) -> str:
        return (
            f"ToolRouter(tools={len(self.registry)}, servers={self.registry.servers!r}, "
            f"embedder={self.embedder.backend!r}, hybrid={self.use_hybrid})"
        )
