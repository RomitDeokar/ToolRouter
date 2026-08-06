"""Tests for the confidence gate -- the most important function in the project.

``ARCHITECTURE.md`` calls out two edge cases explicitly; both are covered here,
along with the core narrow/widen behaviour and the monotonicity property that
makes the ``gap_threshold`` parameter meaningful.
"""

from __future__ import annotations

import pytest

from toolrouter.parser.manifest_parser import Tool
from toolrouter.router.confidence_gate import (
    DEFAULT_GAP_THRESHOLD,
    DEFAULT_SCORE_FLOOR,
    GateMode,
    apply_confidence_gate,
    confidence_gate,
)
from toolrouter.router.retrieve import ScoredTool


def make(name: str, score: float, *, source: str = "dense", **components) -> ScoredTool:
    """Build a ScoredTool with a given score."""
    return ScoredTool(
        tool=Tool(name, f"Tool {name}.", {"properties": {}}, "srv"),
        score=score,
        source=source,
        components=components or {source: score},
    )


# --------------------------------------------------------------------------- #
# Case 1: large gap -> narrow to min_k
# --------------------------------------------------------------------------- #
def test_large_gap_returns_only_min_k():
    candidates = [make("a", 0.95), make("b", 0.60), make("c", 0.55)]
    selected = confidence_gate(candidates, min_k=1, max_k=5, gap_threshold=0.15)
    assert [c.tool.name for c in selected] == ["a"]


def test_large_gap_honours_min_k_above_one():
    candidates = [make("a", 0.95), make("b", 0.60), make("c", 0.58), make("d", 0.57)]
    selected = confidence_gate(candidates, min_k=2, max_k=5, gap_threshold=0.15)
    assert [c.tool.name for c in selected] == ["a", "b"]


def test_gap_exactly_at_threshold_is_confident():
    """The contract is ``gap >= threshold`` -- the boundary counts as confident."""
    candidates = [make("a", 0.80), make("b", 0.65)]
    decision = apply_confidence_gate(
        candidates, min_k=1, max_k=5, gap_threshold=0.15, score_floor=0.0
    )
    assert decision.mode == GateMode.CONFIDENT
    assert len(decision.selected) == 1


# --------------------------------------------------------------------------- #
# Case 2: small gap -> widen to max_k
# --------------------------------------------------------------------------- #
def test_small_gap_returns_up_to_max_k():
    candidates = [make("a", 0.90), make("b", 0.88), make("c", 0.86), make("d", 0.85)]
    selected = confidence_gate(candidates, min_k=1, max_k=3, gap_threshold=0.15)
    assert [c.tool.name for c in selected] == ["a", "b", "c"]


def test_widening_capped_by_available_candidates():
    candidates = [make("a", 0.90), make("b", 0.89)]
    selected = confidence_gate(candidates, min_k=1, max_k=5, gap_threshold=0.15)
    assert len(selected) == 2, "cannot invent candidates that were never retrieved"


def test_ambiguous_mode_is_flagged_as_widened():
    decision = apply_confidence_gate(
        [make("a", 0.90), make("b", 0.89), make("c", 0.88)],
        min_k=1, max_k=5, gap_threshold=0.15, score_floor=0.0,
    )
    assert decision.mode == GateMode.AMBIGUOUS
    assert decision.widened is True


# --------------------------------------------------------------------------- #
# Case 3 (edge case from ARCHITECTURE.md): fewer than 2 candidates
# --------------------------------------------------------------------------- #
def test_single_candidate_does_not_crash():
    """No top1-top2 gap is computable -- must degrade gracefully."""
    decision = apply_confidence_gate([make("solo", 0.90)], score_floor=0.0)
    assert decision.mode == GateMode.SINGLE_CANDIDATE
    assert [c.tool.name for c in decision.selected] == ["solo"]
    assert decision.gap is None, "a gap must not be fabricated from one candidate"
    assert decision.runner_up_score is None


def test_empty_candidate_list_returns_empty():
    decision = apply_confidence_gate([])
    assert decision.mode == GateMode.NO_CANDIDATES
    assert decision.selected == []
    assert decision.has_match is False


def test_empty_candidate_list_via_public_api():
    assert confidence_gate([]) == []


def test_single_candidate_below_floor_is_rejected():
    """The floor still applies when there is only one candidate."""
    decision = apply_confidence_gate([make("solo", 0.10)], score_floor=0.5)
    assert decision.mode == GateMode.NO_CONFIDENT_MATCH
    assert decision.selected == []


# --------------------------------------------------------------------------- #
# Case 4 (edge case from ARCHITECTURE.md): all scores below the floor
# --------------------------------------------------------------------------- #
def test_all_scores_below_floor_returns_no_confident_match():
    """Must not force a top-1 guess when nothing actually matched."""
    candidates = [make("a", 0.20), make("b", 0.19), make("c", 0.18)]
    decision = apply_confidence_gate(candidates, score_floor=0.55)
    assert decision.mode == GateMode.NO_CONFIDENT_MATCH
    assert decision.selected == []
    assert decision.has_match is False


def test_all_scores_below_floor_public_api_returns_empty_list():
    assert confidence_gate([make("a", 0.2), make("b", 0.1)], score_floor=0.55) == []


def test_floor_checked_before_gap_so_bad_matches_are_not_called_ambiguous():
    """Two equally-terrible candidates have a tiny gap.

    Reporting that as "ambiguous" would imply the right answer is in the list.
    It is not -- the honest answer is "no confident match".
    """
    decision = apply_confidence_gate(
        [make("a", 0.20), make("b", 0.199)], score_floor=0.55, gap_threshold=0.15
    )
    assert decision.mode == GateMode.NO_CONFIDENT_MATCH


def test_floor_of_zero_disables_the_check():
    decision = apply_confidence_gate([make("a", 0.01), make("b", 0.005)], score_floor=0.0)
    assert decision.mode != GateMode.NO_CONFIDENT_MATCH
    assert decision.selected


def test_reason_explains_the_floor_rejection():
    decision = apply_confidence_gate([make("a", 0.2)], score_floor=0.55)
    assert "below the absolute floor" in decision.reason
    assert "0.55" in decision.reason


def test_hybrid_floor_uses_raw_dense_score_not_normalised():
    """Hybrid fusion pins the best candidate at 1.0 by construction.

    A normalised score cannot be compared against an absolute floor, so the gate
    must fall back to the pre-normalisation dense score. Without this, no hybrid
    query could ever be rejected as out-of-domain.
    """
    weak = make("a", 1.0, source="hybrid", dense=1.0, bm25=1.0, dense_raw=0.11, bm25_raw=0.4)
    other = make("b", 0.2, source="hybrid", dense=0.0, bm25=0.2, dense_raw=0.09, bm25_raw=0.1)
    decision = apply_confidence_gate([weak, other], score_floor=0.55)
    assert decision.mode == GateMode.NO_CONFIDENT_MATCH, (
        "a normalised 1.0 must not bypass the absolute floor"
    )


# --------------------------------------------------------------------------- #
# Ordering, validation, and properties
# --------------------------------------------------------------------------- #
def test_unsorted_input_is_sorted_defensively():
    candidates = [make("b", 0.60), make("a", 0.95), make("c", 0.30)]
    selected = confidence_gate(candidates, gap_threshold=0.15, score_floor=0.0)
    assert [c.tool.name for c in selected] == ["a"]


def test_ties_broken_deterministically_by_name():
    """Reproducible benchmarks require a stable order under equal scores."""
    first = confidence_gate(
        [make("zebra", 0.9), make("alpha", 0.9)], min_k=1, max_k=2,
        gap_threshold=0.15, score_floor=0.0,
    )
    second = confidence_gate(
        [make("alpha", 0.9), make("zebra", 0.9)], min_k=1, max_k=2,
        gap_threshold=0.15, score_floor=0.0,
    )
    assert [c.tool.name for c in first] == [c.tool.name for c in second] == ["alpha", "zebra"]


def test_max_k_smaller_than_min_k_is_clamped():
    decision = apply_confidence_gate(
        [make("a", 0.9), make("b", 0.89)], min_k=3, max_k=1, score_floor=0.0
    )
    assert decision.max_k == 3
    assert len(decision.selected) <= 3


@pytest.mark.parametrize("bad_min_k", [0, -1])
def test_invalid_min_k_raises(bad_min_k):
    with pytest.raises(ValueError, match="min_k must be >= 1"):
        apply_confidence_gate([make("a", 0.9)], min_k=bad_min_k)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_gap_threshold_raises(bad):
    with pytest.raises(ValueError, match="gap_threshold"):
        apply_confidence_gate([make("a", 0.9)], gap_threshold=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_score_floor_raises(bad):
    with pytest.raises(ValueError, match="score_floor"):
        apply_confidence_gate([make("a", 0.9)], score_floor=bad)


def test_selection_count_is_monotonic_in_gap_threshold():
    """Raising the threshold can only widen (never narrow) the selection.

    This is the property that makes the parameter tunable in a predictable
    direction -- if it did not hold, the calibration sweep would be meaningless.
    """
    candidates = [make(chr(97 + i), 0.9 - i * 0.02) for i in range(6)]
    sizes = [
        len(confidence_gate(candidates, min_k=1, max_k=5, gap_threshold=t, score_floor=0.0))
        for t in (0.0, 0.01, 0.05, 0.1, 0.3)
    ]
    assert sizes == sorted(sizes), f"selection size must be non-decreasing, got {sizes}"


def test_decision_serialises_to_json_friendly_dict():
    decision = apply_confidence_gate(
        [make("a", 0.95), make("b", 0.5)], gap_threshold=0.15, score_floor=0.0
    )
    payload = decision.to_dict()
    assert payload["mode"] == GateMode.CONFIDENT
    assert payload["selected_k"] == 1
    assert payload["gap"] == pytest.approx(0.45)
    assert payload["widened"] is False
    assert isinstance(payload["reason"], str) and payload["reason"]


def test_reason_cites_the_actual_numbers():
    decision = apply_confidence_gate(
        [make("winner", 0.95), make("loser", 0.40)], gap_threshold=0.15, score_floor=0.0
    )
    assert "winner" in decision.reason and "loser" in decision.reason
    assert "0.550" in decision.reason or "0.55" in decision.reason


# --------------------------------------------------------------------------- #
# Shipped defaults
# --------------------------------------------------------------------------- #
def test_shipped_defaults_are_the_calibrated_values():
    """Guards against silently reverting to the un-measured illustrative values."""
    assert DEFAULT_GAP_THRESHOLD == 0.03
    assert DEFAULT_SCORE_FLOOR == 0.54


def test_defaults_narrow_on_a_clear_winner():
    decision = apply_confidence_gate([make("a", 0.85), make("b", 0.70), make("c", 0.68)])
    assert decision.mode == GateMode.CONFIDENT
    assert len(decision.selected) == 1


def test_defaults_widen_on_a_near_tie():
    decision = apply_confidence_gate(
        [make("a", 0.80), make("b", 0.795), make("c", 0.79), make("d", 0.70)]
    )
    assert decision.mode == GateMode.AMBIGUOUS
    assert len(decision.selected) > 1
