"""Confidence-gate threshold calibration sweep.

``ARCHITECTURE.md`` proposes ``gap_threshold=0.15`` as an illustrative value. It
is illustrative, not measured -- the right value depends on how tightly the
chosen embedding model packs scores on the actual tool corpus. This module sweeps
the parameter and reports the trade-off so the default is a *measured* choice.

What the sweep optimises
------------------------
The gate has two competing jobs:

* on **clean** queries, narrow to one tool (cheap prompts, no ambiguity), and
* on **ambiguous** queries, widen so the correct tool stays in context.

So the objective is: maximise clean-query narrowing *and* ambiguous-query
recovery simultaneously. We report both columns plus their harmonic mean, and
also the score-floor separation between in-domain and out-of-domain queries.

Run
---
    python -m toolrouter.bench.calibrate
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from ..router.confidence_gate import apply_confidence_gate
from .generate_dataset import load_dataset
from .metrics import mean, percentile

__all__ = ["sweep_gap_threshold", "score_floor_report", "main"]

logger = logging.getLogger(__name__)

#: Candidate gap thresholds to sweep.
GAP_GRID: tuple[float, ...] = (
    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25,
)

#: Deliberately out-of-domain queries. The score floor should reject all of them
#: while accepting genuine in-domain queries from the dataset.
OUT_OF_DOMAIN_QUERIES: tuple[str, ...] = (
    "what is the weather in tokyo tomorrow",
    "solve this differential equation for me",
    "who won the world cup in 1998",
    "translate this paragraph into german",
    "write a python script to parse csv files",
    "what is the capital of peru",
    "book me a flight to london next friday",
    "remind me to call my dentist",
    "summarise this legal contract",
    "what is my current bitcoin balance",
)


def sweep_gap_threshold(
    router,
    dataset_path: str,
    *,
    grid: Sequence[float] = GAP_GRID,
    k: int = 5,
    score_floor: float = 0.55,
) -> list[dict]:
    """Sweep ``gap_threshold`` and report behaviour per value.

    Candidates are retrieved **once per query** and the gate is re-applied for
    each threshold, so the sweep measures only the gate and costs one retrieval
    pass rather than ``len(grid)`` passes.
    """
    dataset = load_dataset(dataset_path)
    cached = [(row, router.retrieve(row.query, k=k, hybrid=False)) for row in dataset]

    clean = [(r, c) for r, c in cached if r.category == "clean"]
    ambiguous = [(r, c) for r, c in cached if r.category == "ambiguous"]

    rows: list[dict] = []
    for threshold in grid:
        # `threshold` is bound as a default argument rather than captured from
        # the enclosing loop: a late-binding closure here would silently
        # evaluate every row against the *last* grid value if this helper were
        # ever stored and called after the loop.
        def gate(candidates, threshold=threshold):
            return apply_confidence_gate(
                candidates,
                min_k=1,
                max_k=k,
                gap_threshold=threshold,
                score_floor=score_floor,
            )

        clean_decisions = [(r, gate(c)) for r, c in clean]
        amb_decisions = [(r, gate(c)) for r, c in ambiguous]
        all_decisions = [(r, gate(c)) for r, c in cached]

        # Clean queries: we want a single tool AND for it to be the right one.
        clean_narrowed = mean(
            [1.0 if d.mode == "confident" else 0.0 for _, d in clean_decisions]
        )
        clean_correct_single = mean(
            [
                1.0
                if len(d.selected) == 1 and d.selected[0].tool.name == r.correct_tool
                else 0.0
                for r, d in clean_decisions
            ]
        )
        # Ambiguous queries: we want the correct tool to survive in context.
        amb_widened = mean([1.0 if d.widened else 0.0 for _, d in amb_decisions])
        amb_in_context = mean(
            [
                1.0 if r.correct_tool in [s.tool.name for s in d.selected] else 0.0
                for r, d in amb_decisions
            ]
        )
        avg_context = mean([float(len(d.selected)) for _, d in all_decisions])
        overall_in_context = mean(
            [
                1.0 if r.correct_tool in [s.tool.name for s in d.selected] else 0.0
                for r, d in all_decisions
            ]
        )

        # Harmonic mean balances the two competing goals; a threshold that wins
        # one at the cost of the other scores poorly here.
        if clean_correct_single + amb_in_context > 0:
            balance = (
                2 * clean_correct_single * amb_in_context
                / (clean_correct_single + amb_in_context)
            )
        else:
            balance = 0.0

        rows.append(
            {
                "gap_threshold": threshold,
                "clean_narrow_rate": clean_narrowed,
                "clean_correct_as_single": clean_correct_single,
                "ambiguous_widen_rate": amb_widened,
                "ambiguous_correct_in_context": amb_in_context,
                "overall_correct_in_context": overall_in_context,
                "avg_tools_in_context": avg_context,
                "balance_score": balance,
            }
        )
    return rows


def score_floor_report(router, dataset_path: str, *, k: int = 5) -> dict:
    """Compare in-domain vs out-of-domain top-1 scores to justify the floor."""
    dataset = load_dataset(dataset_path)
    in_domain = []
    for row in dataset:
        candidates = router.retrieve(row.query, k=k, hybrid=False)
        if candidates:
            in_domain.append(float(candidates[0].score))

    out_domain = []
    for query in OUT_OF_DOMAIN_QUERIES:
        candidates = router.retrieve(query, k=k, hybrid=False)
        if candidates:
            out_domain.append(float(candidates[0].score))

    return {
        "in_domain": {
            "n": len(in_domain),
            "min": min(in_domain) if in_domain else 0.0,
            "p05": percentile(in_domain, 5),
            "p25": percentile(in_domain, 25),
            "median": percentile(in_domain, 50),
            "max": max(in_domain) if in_domain else 0.0,
        },
        "out_of_domain": {
            "n": len(out_domain),
            "min": min(out_domain) if out_domain else 0.0,
            "median": percentile(out_domain, 50),
            "p95": percentile(out_domain, 95),
            "max": max(out_domain) if out_domain else 0.0,
        },
    }


def render_report(sweep: list[dict], floor: dict, *, chosen: float, stats: dict) -> str:
    """Render the calibration markdown report."""
    lines = [
        "# Confidence gate calibration",
        "",
        "Generated by `python -m toolrouter.bench.calibrate` -- do not edit by hand.",
        "",
        "`ARCHITECTURE.md` proposes `gap_threshold=0.15` as an illustrative value.",
        "This sweep measures what that parameter actually does on this corpus with",
        f"`{stats.get('embedder')}`, so the shipped default is measured rather than assumed.",
        "",
        "## Gap threshold sweep",
        "",
        "| gap_threshold | Clean: narrowed to 1 | Clean: correct as the single tool "
        "| Ambiguous: widened | Ambiguous: correct in context | Overall: correct in context "
        "| Avg tools in context | Balance |",
        "|---|---|---|---|---|---|---|---|",
    ]
    best = max(sweep, key=lambda r: r["balance_score"]) if sweep else None
    for row in sweep:
        marker = ""
        if abs(row["gap_threshold"] - chosen) < 1e-9:
            marker = " **<- shipped default**"
        elif best and abs(row["gap_threshold"] - best["gap_threshold"]) < 1e-9:
            marker = " *(best balance)*"
        lines.append(
            f"| {row['gap_threshold']:.2f}{marker} "
            f"| {row['clean_narrow_rate'] * 100:.1f}% "
            f"| {row['clean_correct_as_single'] * 100:.1f}% "
            f"| {row['ambiguous_widen_rate'] * 100:.1f}% "
            f"| {row['ambiguous_correct_in_context'] * 100:.1f}% "
            f"| {row['overall_correct_in_context'] * 100:.1f}% "
            f"| {row['avg_tools_in_context']:.2f} "
            f"| {row['balance_score']:.3f} |"
        )

    lines += [
        "",
        "**Balance** is the harmonic mean of *clean: correct as the single tool* and",
        "*ambiguous: correct in context* -- the gate's two competing jobs. A threshold",
        "that wins one at the other's expense scores poorly.",
        "",
        "Reading the sweep: at `0.00` the gate never widens (it degenerates into",
        "fixed top-1), and at `0.15`+ it widens almost everywhere (it degenerates into",
        "fixed top-k). Both extremes throw away the point of adaptive-k. The useful",
        "range on this corpus is where the two columns are simultaneously high.",
        "",
        "## Score floor separation",
        "",
        "The absolute floor decides when to return *nothing* rather than a forced guess.",
        "It only works if in-domain and out-of-domain top-1 scores actually separate:",
        "",
        "| Query set | n | min | p05 | p25 | median | p95 | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    in_d, out_d = floor["in_domain"], floor["out_of_domain"]
    lines.append(
        f"| In-domain (dataset) | {in_d['n']} | {in_d['min']:.3f} | {in_d['p05']:.3f} "
        f"| {in_d['p25']:.3f} | {in_d['median']:.3f} | -- | {in_d['max']:.3f} |"
    )
    lines.append(
        f"| Out-of-domain (held-out) | {out_d['n']} | {out_d['min']:.3f} | -- | -- "
        f"| {out_d['median']:.3f} | {out_d['p95']:.3f} | {out_d['max']:.3f} |"
    )
    lines += [
        "",
        f"Out-of-domain queries top out at **{out_d['max']:.3f}**, while the in-domain "
        f"5th percentile is **{in_d['p05']:.3f}**.",
        "",
        "Out-of-domain queries used for this measurement (none are servable by any tool",
        "in the manifest):",
        "",
    ]
    for query in OUT_OF_DOMAIN_QUERIES:
        lines.append(f"- `{query}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from .. import ToolRouter
    from ..router.confidence_gate import DEFAULT_GAP_THRESHOLD, DEFAULT_SCORE_FLOOR

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Calibrate the confidence gate.")
    parser.add_argument("--manifest", default="examples/swiggy_manifest.json")
    parser.add_argument("--dataset", default="toolrouter/bench/dataset.jsonl")
    parser.add_argument("--out", default="bench_results/calibration.md")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    router = ToolRouter.from_manifest(args.manifest)
    sweep = sweep_gap_threshold(
        router, args.dataset, k=args.k, score_floor=DEFAULT_SCORE_FLOOR
    )
    floor = score_floor_report(router, args.dataset, k=args.k)
    report = render_report(
        sweep, floor, chosen=DEFAULT_GAP_THRESHOLD, stats=router.stats()
    )

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    with target.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump({"sweep": sweep, "score_floor": floor}, handle, indent=2)

    print(report)
    print(f"Wrote {target}")


if __name__ == "__main__":  # pragma: no cover
    main()
