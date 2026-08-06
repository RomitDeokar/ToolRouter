"""The four CommerceBench baselines -- exactly four, per ``BENCHMARK.md``.

Each function has the same signature and returns a :class:`BaselineOutcome`, so
:mod:`toolrouter.bench.evaluate` can score them uniformly.

1. ``all_tools``       -- no retrieval; every tool schema goes into context.
2. ``dense``           -- embedding similarity, fixed top-k.
3. ``hybrid``          -- normalised dense + BM25 fusion, fixed top-k.
4. ``confidence_gate`` -- dense retrieval with adaptive-k gating.

A note on how the ``all_tools`` baseline is scored
--------------------------------------------------
"All tools" has no ranking of its own, so scoring it needs an explicit and
honest convention rather than a silent one. Its tool list is the registry in
**manifest order**, which means:

* ``Recall@k`` for k >= N is 1.0 by construction -- the correct tool is always
  somewhere in context.
* ``Top-1``/``Top-3``/``MRR`` reflect manifest position, which carries no
  semantic signal. We report them, but the honest reading is "this baseline
  provides no ranking"; its real cost shows up in the prompt-token column.

This is exactly why ``BENCHMARK.md`` insists the baseline gets a real measured
number instead of an assumption: the interesting comparison is tokens and
selection burden, not a fake ranking win.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..parser.manifest_parser import Tool
from ..router.prompt_builder import build_tool_prompt, estimate_tokens

if TYPE_CHECKING:  # pragma: no cover
    from .. import ToolRouter

__all__ = [
    "BaselineOutcome",
    "BASELINES",
    "run_all_tools",
    "run_dense",
    "run_hybrid",
    "run_confidence_gate",
    "run_baseline",
]

#: Canonical baseline names, in report order.
BASELINES: tuple[str, ...] = ("all_tools", "dense", "hybrid", "confidence_gate")

#: Prompt rendering per baseline. The unrouted baseline is measured with full
#: JSON Schema because that is what agent frameworks actually serialise; the
#: routed baselines use the compact rendering the prompt builder emits.
_PROMPT_STYLE = {
    "all_tools": "json",
    "dense": "compact",
    "hybrid": "compact",
    "confidence_gate": "compact",
}


@dataclass
class BaselineOutcome:
    """One baseline's response to one query."""

    query: str
    ranked_tools: list[str]  # ordered tool names, best first
    prompt_tokens: int  # tokens spent on tool schemas
    latency_ms: float  # retrieval wall-clock time
    tools_in_context: int
    gate_mode: str | None = None
    gate_widened: bool = False
    scores: list[float] = field(default_factory=list)

    @property
    def top_tool(self) -> str | None:
        return self.ranked_tools[0] if self.ranked_tools else None


def _tool_names(tools: list[Tool]) -> list[str]:
    return [t.name for t in tools]


# --------------------------------------------------------------------------- #
# Baseline 1 -- no retrieval
# --------------------------------------------------------------------------- #
def run_all_tools(router: ToolRouter, query: str, k: int = 5) -> BaselineOutcome:
    """Every tool in context, in manifest order. The thing we're improving on."""
    start = time.perf_counter()
    tools = router.registry.tools  # no work done -- that is the point
    latency_ms = (time.perf_counter() - start) * 1000.0

    prompt = build_tool_prompt(tools, style=_PROMPT_STYLE["all_tools"], header=False)
    return BaselineOutcome(
        query=query,
        ranked_tools=_tool_names(tools),
        prompt_tokens=estimate_tokens(prompt),
        latency_ms=latency_ms,
        tools_in_context=len(tools),
        gate_mode=None,
        gate_widened=False,
        scores=[0.0] * len(tools),
    )


# --------------------------------------------------------------------------- #
# Baselines 2 & 3 -- fixed top-k retrieval
# --------------------------------------------------------------------------- #
def _fixed_topk(
    router: ToolRouter, query: str, k: int, *, hybrid: bool, label: str
) -> BaselineOutcome:
    start = time.perf_counter()
    candidates = router.retrieve(query, k=k, hybrid=hybrid)
    latency_ms = (time.perf_counter() - start) * 1000.0

    tools = [c.tool for c in candidates]
    prompt = build_tool_prompt(tools, style=_PROMPT_STYLE[label], header=False) if tools else ""
    return BaselineOutcome(
        query=query,
        ranked_tools=_tool_names(tools),
        prompt_tokens=estimate_tokens(prompt),
        latency_ms=latency_ms,
        tools_in_context=len(tools),
        gate_mode=None,
        gate_widened=False,
        scores=[float(c.score) for c in candidates],
    )


def run_dense(router: ToolRouter, query: str, k: int = 5) -> BaselineOutcome:
    """Dense embedding retrieval, fixed top-k."""
    return _fixed_topk(router, query, k, hybrid=False, label="dense")


def run_hybrid(router: ToolRouter, query: str, k: int = 5) -> BaselineOutcome:
    """Normalised dense + BM25 fusion, fixed top-k.

    Requires the router to have been built with ``use_hybrid=True``.
    """
    if router.bm25 is None:
        raise RuntimeError(
            "The hybrid baseline needs a BM25 index. Build the router with "
            "ToolRouter.from_manifest(..., use_hybrid=True)."
        )
    return _fixed_topk(router, query, k, hybrid=True, label="hybrid")


# --------------------------------------------------------------------------- #
# Baseline 4 -- adaptive-k
# --------------------------------------------------------------------------- #
def run_confidence_gate(router: ToolRouter, query: str, k: int = 5) -> BaselineOutcome:
    """Dense retrieval plus the adaptive-k confidence gate.

    ``max_k`` is set to the same ``k`` the fixed-top-k baselines use, so the
    comparison isolates the *gate's* effect rather than confounding it with a
    different candidate budget.
    """
    result = router.route(query, k=k, hybrid=False, use_gate=True, min_k=1, max_k=k)
    prompt = (
        build_tool_prompt(
            result.tools, style=_PROMPT_STYLE["confidence_gate"], header=False
        )
        if result.tools
        else ""
    )
    return BaselineOutcome(
        query=query,
        ranked_tools=result.tool_names,
        prompt_tokens=estimate_tokens(prompt),
        latency_ms=result.latency_ms,
        tools_in_context=len(result.tools),
        gate_mode=result.gate.get("mode"),
        gate_widened=bool(result.gate.get("widened")),
        scores=[float(s.score) for s in result.scored],
    )


_RUNNERS = {
    "all_tools": run_all_tools,
    "dense": run_dense,
    "hybrid": run_hybrid,
    "confidence_gate": run_confidence_gate,
}


def run_baseline(
    baseline: str, router: ToolRouter, query: str, k: int = 5
) -> BaselineOutcome:
    """Dispatch to a baseline by name."""
    try:
        runner = _RUNNERS[baseline]
    except KeyError:
        raise ValueError(
            f"Unknown baseline {baseline!r}. Expected one of {list(BASELINES)}."
        ) from None
    return runner(router, query, k=k)
