"""The confidence gate -- adaptive-k selection.

This is the function that turns *retrieval* into *routing*. Fixed top-k always
returns k tools whether the answer is obvious or genuinely ambiguous. The gate
instead reads the **score gap** between the top two candidates:

* **large gap** -> one candidate clearly wins -> return ``min_k`` (narrow, cheap,
  unambiguous for the LLM);
* **small gap** -> several candidates are plausible -> return up to ``max_k`` and
  let the LLM disambiguate with more context;
* **everything below an absolute floor** -> nothing actually matched -> report
  ``no_confident_match`` instead of forcing a top-1 guess.

That last branch matters: a router that always answers is indistinguishable from
a router that has no idea, and forcing a guess on an out-of-domain query is how
agents end up calling ``place_order`` when the user asked about the weather.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .retrieve import ScoredTool

__all__ = [
    "GateDecision",
    "GateMode",
    "confidence_gate",
    "apply_confidence_gate",
    "DEFAULT_GAP_THRESHOLD",
    "DEFAULT_SCORE_FLOOR",
]

logger = logging.getLogger(__name__)

#: Minimum top1-top2 score gap to be considered "confident".
#:
#: ``ARCHITECTURE.md`` offers 0.15 as an illustrative value. Measured against real
#: ``bge-small-en-v1.5`` cosine scores on this tool corpus, 0.15 is far too strict:
#: it narrows only 12.5% of clean queries, so the gate degenerates into fixed
#: top-k and adaptive-k stops meaning anything. 0.03 is the calibrated optimum
#: (86.5% of clean queries narrowed to a single correct tool, while 100% of
#: ambiguous queries keep the correct tool in context). Reproduce with
#: ``python -m toolrouter.bench.calibrate``; the sweep is in
#: ``bench_results/calibration.md``.
DEFAULT_GAP_THRESHOLD = 0.03
#: Minimum absolute top-1 cosine for any match to be considered real at all.
#:
#: The in-domain and out-of-domain score distributions overlap on this corpus
#: (in-domain min 0.543, out-of-domain max 0.583), so no floor separates them
#: perfectly and the choice is an explicit error trade-off. The two errors are
#: not symmetric:
#:
#: * rejecting a valid query is a **hard** failure -- the user is told outright
#:   that no tool can serve them, and there is no recovery path;
#: * failing to reject an out-of-domain query is a **soft** failure -- the LLM
#:   receives some tools, sees none of them fit, and can still decline.
#:
#: So the floor is set to the highest value that rejects *zero* valid dataset
#: queries: 0.54 (measured 0.0% in-domain rejection, catching 6/10 held-out
#: out-of-domain queries). Raising it to 0.59 catches all 10 but starts
#: rejecting 4.7% of valid queries, 6 of which would otherwise have been
#: answered correctly -- a bad trade given the asymmetry above.
#:
#: This is corpus- and model-specific. Re-run ``python -m toolrouter.bench.calibrate``
#: after changing the manifest or the embedding model.
DEFAULT_SCORE_FLOOR = 0.54


class GateMode:
    """Enumeration of gate outcomes (plain strings -- they end up in JSON)."""

    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    SINGLE_CANDIDATE = "single_candidate"
    NO_CANDIDATES = "no_candidates"
    NO_CONFIDENT_MATCH = "no_confident_match"


@dataclass
class GateDecision:
    """Why the gate returned what it returned -- the auditable half of routing."""

    mode: str
    selected: list[ScoredTool] = field(default_factory=list)
    gap: float | None = None
    top_score: float | None = None
    runner_up_score: float | None = None
    gap_threshold: float = DEFAULT_GAP_THRESHOLD
    score_floor: float = DEFAULT_SCORE_FLOOR
    min_k: int = 1
    max_k: int = 5
    reason: str = ""

    @property
    def widened(self) -> bool:
        """``True`` when the gate returned more than ``min_k`` due to ambiguity."""
        return self.mode == GateMode.AMBIGUOUS

    @property
    def has_match(self) -> bool:
        return bool(self.selected)

    def to_dict(self) -> dict:
        def _round(value: float | None) -> float | None:
            return None if value is None else round(float(value), 6)

        return {
            "mode": self.mode,
            "selected_k": len(self.selected),
            "gap": _round(self.gap),
            "top_score": _round(self.top_score),
            "runner_up_score": _round(self.runner_up_score),
            "gap_threshold": _round(self.gap_threshold),
            "score_floor": _round(self.score_floor),
            "min_k": self.min_k,
            "max_k": self.max_k,
            "widened": self.widened,
            "reason": self.reason,
        }


def _floor_score(candidate: ScoredTool) -> float:
    """Score used for the absolute-floor test.

    Hybrid fusion min-max normalises its inputs, which pins the best candidate at
    exactly ``1.0`` regardless of how weak the match actually was -- so a
    normalised score can't be compared against an absolute floor. When the raw
    pre-normalisation dense score is available we use that instead; otherwise we
    fall back to the candidate's score.
    """
    raw = candidate.components.get("dense_raw")
    if raw is not None:
        return float(raw)
    return float(candidate.score)


def apply_confidence_gate(
    candidates: list[ScoredTool],
    min_k: int = 1,
    max_k: int = 5,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    score_floor: float = DEFAULT_SCORE_FLOOR,
) -> GateDecision:
    """Run the gate and return the full :class:`GateDecision`.

    :func:`confidence_gate` is the thin contract-conforming wrapper around this;
    use this variant when you want the decision metadata too.

    Parameters
    ----------
    candidates:
        Ranked candidates, best first. Re-sorted defensively.
    min_k:
        How many tools to return when confident. Must be >= 1.
    max_k:
        Upper bound when ambiguous. Clamped up to ``min_k`` if smaller.
    gap_threshold:
        Score gap at or above which the gate is confident.
    score_floor:
        Absolute score the best candidate must reach for *any* match to be
        reported. Pass ``0.0`` to disable the floor entirely.
    """
    if min_k < 1:
        raise ValueError(f"min_k must be >= 1, got {min_k}.")
    if max_k < min_k:
        logger.debug("max_k (%d) < min_k (%d); clamping max_k to min_k.", max_k, min_k)
        max_k = min_k
    if not 0.0 <= gap_threshold <= 1.0:
        raise ValueError(f"gap_threshold must be in [0, 1], got {gap_threshold}.")
    if not 0.0 <= score_floor <= 1.0:
        raise ValueError(f"score_floor must be in [0, 1], got {score_floor}.")

    base = {
        "gap_threshold": gap_threshold,
        "score_floor": score_floor,
        "min_k": min_k,
        "max_k": max_k,
    }

    # -- Edge case: nothing retrieved at all.
    if not candidates:
        return GateDecision(
            mode=GateMode.NO_CANDIDATES,
            selected=[],
            reason="Retrieval returned no candidates; nothing to gate.",
            **base,
        )

    ranked = sorted(candidates, key=lambda c: (-float(c.score), c.tool.name))
    top = ranked[0]
    top_score = float(top.score)

    # -- Edge case: no candidate clears the absolute floor.
    # Checked before the gap test on purpose: two equally-terrible candidates
    # produce a tiny gap, and reporting that as "ambiguous" would imply the right
    # answer is somewhere in the list when it probably isn't there at all.
    if score_floor > 0.0 and _floor_score(top) < score_floor:
        return GateDecision(
            mode=GateMode.NO_CONFIDENT_MATCH,
            selected=[],
            gap=None,
            top_score=top_score,
            runner_up_score=float(ranked[1].score) if len(ranked) > 1 else None,
            reason=(
                f"Best candidate {top.tool.name!r} scored "
                f"{_floor_score(top):.3f}, below the absolute floor {score_floor:.2f}. "
                "No tool in the index plausibly serves this query, so no guess is "
                "returned."
            ),
            **base,
        )

    # -- Edge case: a single candidate -- no gap is computable.
    if len(ranked) < 2:
        return GateDecision(
            mode=GateMode.SINGLE_CANDIDATE,
            selected=ranked[:min_k],
            gap=None,
            top_score=top_score,
            runner_up_score=None,
            reason=(
                f"Only one candidate ({top.tool.name!r}, score {top_score:.3f}) was "
                "retrieved, so no top1-top2 gap exists; returning it as-is."
            ),
            **base,
        )

    runner_up_score = float(ranked[1].score)
    gap = top_score - runner_up_score

    if gap >= gap_threshold:
        selected = ranked[:min_k]
        return GateDecision(
            mode=GateMode.CONFIDENT,
            selected=selected,
            gap=gap,
            top_score=top_score,
            runner_up_score=runner_up_score,
            reason=(
                f"Gap {gap:.3f} between {top.tool.name!r} ({top_score:.3f}) and "
                f"{ranked[1].tool.name!r} ({runner_up_score:.3f}) meets the threshold "
                f"{gap_threshold:.2f}; narrowing to top-{len(selected)}."
            ),
            **base,
        )

    selected = ranked[:max_k]
    return GateDecision(
        mode=GateMode.AMBIGUOUS,
        selected=selected,
        gap=gap,
        top_score=top_score,
        runner_up_score=runner_up_score,
        reason=(
            f"Gap {gap:.3f} between {top.tool.name!r} ({top_score:.3f}) and "
            f"{ranked[1].tool.name!r} ({runner_up_score:.3f}) is below the threshold "
            f"{gap_threshold:.2f}; widening to top-{len(selected)} so the LLM can "
            "disambiguate."
        ),
        **base,
    )


def confidence_gate(
    candidates: list[ScoredTool],
    min_k: int = 1,
    max_k: int = 5,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    score_floor: float = DEFAULT_SCORE_FLOOR,
) -> list[ScoredTool]:
    """Adaptive-k filter over ranked candidates.

    Returns ``candidates[:min_k]`` when the top-1/top-2 score gap is at least
    ``gap_threshold``, ``candidates[:max_k]`` when it is smaller, and ``[]`` when
    no candidate clears ``score_floor``.

    Examples
    --------
    >>> from toolrouter.parser.manifest_parser import Tool
    >>> from toolrouter.router.retrieve import ScoredTool
    >>> mk = lambda n, s: ScoredTool(Tool(n, "d", {}, "srv"), s, "dense")
    >>> [c.tool.name for c in confidence_gate([mk("a", 0.95), mk("b", 0.60)])]
    ['a']
    >>> len(confidence_gate([mk("a", 0.90), mk("b", 0.88), mk("c", 0.86)]))
    3
    >>> confidence_gate([mk("a", 0.20), mk("b", 0.19)])
    []
    """
    return apply_confidence_gate(
        candidates,
        min_k=min_k,
        max_k=max_k,
        gap_threshold=gap_threshold,
        score_floor=score_floor,
    ).selected
