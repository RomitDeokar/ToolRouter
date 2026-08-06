"""Retrieval, adaptive-k gating, explainability, and prompt construction."""

from .confidence_gate import (
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_SCORE_FLOOR,
    GateDecision,
    GateMode,
    apply_confidence_gate,
    confidence_gate,
)
from .explain import explain_candidates
from .prompt_builder import (
    build_no_match_prompt,
    build_tool_prompt,
    estimate_tokens,
    tokens_for_tools,
)
from .retrieve import DEFAULT_DENSE_WEIGHT, Retriever, RouteResult, ScoredTool

__all__ = [
    "DEFAULT_DENSE_WEIGHT",
    "DEFAULT_GAP_THRESHOLD",
    "DEFAULT_SCORE_FLOOR",
    "GateDecision",
    "GateMode",
    "Retriever",
    "RouteResult",
    "ScoredTool",
    "apply_confidence_gate",
    "build_no_match_prompt",
    "build_tool_prompt",
    "confidence_gate",
    "estimate_tokens",
    "explain_candidates",
    "tokens_for_tools",
]
