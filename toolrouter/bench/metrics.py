"""Standard IR metrics -- no invented ones (per ``BENCHMARK.md``).

Every metric here takes a ranked list of tool names plus the single ground-truth
tool name, matching the dataset format. Definitions are the textbook ones so the
numbers are comparable to any other retrieval evaluation:

* **Top-n accuracy** -- is the correct tool within the first n positions?
* **Recall@k** -- with exactly one relevant item per query, this equals
  Top-k accuracy. Both are reported because ``BENCHMARK.md`` asks for both;
  they are *not* independent signals, and the summary says so.
* **MRR** -- ``1 / rank`` of the correct tool, else 0.
* **NDCG@k** -- with one binary-relevant item, ``1 / log2(rank + 1)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "rank_of",
    "top_n_accuracy",
    "reciprocal_rank",
    "recall_at_k",
    "ndcg_at_k",
    "percentile",
    "mean",
]


def rank_of(ranked: Sequence[str], correct: str) -> int | None:
    """1-based rank of ``correct`` in ``ranked``, or ``None`` if absent."""
    for index, name in enumerate(ranked, start=1):
        if name == correct:
            return index
    return None


def top_n_accuracy(ranked: Sequence[str], correct: str, n: int) -> float:
    """1.0 if the correct tool appears in the first ``n`` positions."""
    rank = rank_of(ranked, correct)
    return 1.0 if rank is not None and rank <= n else 0.0


def reciprocal_rank(ranked: Sequence[str], correct: str) -> float:
    """``1 / rank`` of the correct tool; 0.0 when it was not retrieved."""
    rank = rank_of(ranked, correct)
    return 1.0 / rank if rank else 0.0


def recall_at_k(ranked: Sequence[str], correct: str, k: int) -> float:
    """Recall@k. Identical to Top-k accuracy for single-relevant-item queries."""
    return top_n_accuracy(ranked, correct, k)


def ndcg_at_k(ranked: Sequence[str], correct: str, k: int) -> float:
    """NDCG@k for one binary-relevant item.

    DCG is ``1 / log2(rank + 1)``; the ideal DCG (correct item at rank 1) is
    ``1.0``, so NDCG reduces to the DCG term itself.
    """
    rank = rank_of(ranked, correct)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence."""
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in ``[0, 100]``).

    Implemented locally rather than via ``numpy.percentile`` so the metrics
    module has no array dependency and the arithmetic is auditable.
    """
    if not values:
        return 0.0
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
