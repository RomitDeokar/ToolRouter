"""CommerceBench evaluation harness.

Implements the ``evaluate()`` contract from ``BENCHMARK.md``: run one baseline
over the dataset, compute the standard metric set plus a per-category breakdown,
write ``bench_results/<baseline>.json``, and append a row to
``bench_results/summary.md``.

The summary table is the file the README's Results section is populated from --
numbers are never hand-transcribed.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..router.prompt_builder import build_tool_prompt, estimate_tokens
from .baselines import BASELINES, BaselineOutcome, run_baseline
from .generate_dataset import BenchQuery, load_dataset
from .metrics import mean, ndcg_at_k, percentile, rank_of, recall_at_k, reciprocal_rank

if TYPE_CHECKING:  # pragma: no cover
    from .. import ToolRouter

__all__ = ["evaluate", "evaluate_all", "DEFAULT_DATASET", "DEFAULT_OUTPUT_DIR"]

logger = logging.getLogger(__name__)

DEFAULT_DATASET = "toolrouter/bench/dataset.jsonl"
DEFAULT_OUTPUT_DIR = "bench_results"

#: k values reported for Recall@k.
RECALL_KS = (1, 3, 5)


def _tokenizer_name() -> str:
    try:  # pragma: no cover - optional dependency
        import tiktoken  # noqa: F401

        return "tiktoken:cl100k_base"
    except Exception:  # noqa: BLE001
        return "heuristic:~4-chars-per-token"


def _aggregate(
    rows: Sequence[tuple[BenchQuery, BaselineOutcome]], *, baseline: str
) -> dict:
    """Compute the metric block for a set of (query, outcome) pairs."""
    if not rows:
        return {"queries": 0}

    ranks = [rank_of(o.ranked_tools, q.correct_tool) for q, o in rows]
    latencies = [o.latency_ms for _, o in rows]
    tokens = [o.prompt_tokens for _, o in rows]
    context_sizes = [o.tools_in_context for _, o in rows]

    metrics: dict = {
        "queries": len(rows),
        "top_1_accuracy": mean([1.0 if r == 1 else 0.0 for r in ranks]),
        "top_3_accuracy": mean([1.0 if r is not None and r <= 3 else 0.0 for r in ranks]),
        "mrr": mean([reciprocal_rank(o.ranked_tools, q.correct_tool) for q, o in rows]),
        "ndcg_at_5": mean([ndcg_at_k(o.ranked_tools, q.correct_tool, 5) for q, o in rows]),
        "avg_prompt_tokens": mean([float(t) for t in tokens]),
        "avg_tools_in_context": mean([float(c) for c in context_sizes]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": mean(latencies),
    }
    for k in RECALL_KS:
        metrics[f"recall_at_{k}"] = mean(
            [recall_at_k(o.ranked_tools, q.correct_tool, k) for q, o in rows]
        )

    # Gate behaviour -- only meaningful for the gated baseline.
    gate_modes: dict[str, int] = {}
    for _, outcome in rows:
        if outcome.gate_mode:
            gate_modes[outcome.gate_mode] = gate_modes.get(outcome.gate_mode, 0) + 1
    if gate_modes:
        metrics["gate_modes"] = gate_modes
        metrics["gate_widen_rate"] = mean([1.0 if o.gate_widened else 0.0 for _, o in rows])
        metrics["gate_no_match_rate"] = mean(
            [1.0 if o.gate_mode == "no_confident_match" else 0.0 for _, o in rows]
        )
    return metrics


def _ambiguous_behaviour(
    rows: Sequence[tuple[BenchQuery, BaselineOutcome]]
) -> dict:
    """The metric nobody else reports: what does the router do when unsure?

    On ``ambiguous`` rows, a fixed-top-k retriever always returns k tools with no
    signal about its own uncertainty. The gate should instead *widen* -- and
    crucially, widening should convert would-be top-1 misses into "correct tool
    is still in context" wins, so the LLM can recover. We measure exactly that.
    """
    ambiguous = [(q, o) for q, o in rows if q.category == "ambiguous"]
    if not ambiguous:
        return {"ambiguous_queries": 0}

    widened = [o for _, o in ambiguous if o.gate_widened]
    top1_hits = sum(
        1 for q, o in ambiguous if o.ranked_tools and o.ranked_tools[0] == q.correct_tool
    )
    in_context = sum(1 for q, o in ambiguous if q.correct_tool in o.ranked_tools)

    # Of the ambiguous queries the router got wrong at rank 1, how many were
    # nonetheless "rescued" by having the correct tool somewhere in context?
    missed_top1 = [
        (q, o)
        for q, o in ambiguous
        if not (o.ranked_tools and o.ranked_tools[0] == q.correct_tool)
    ]
    rescued = sum(1 for q, o in missed_top1 if q.correct_tool in o.ranked_tools)

    return {
        "ambiguous_queries": len(ambiguous),
        "ambiguous_widen_rate": len(widened) / len(ambiguous),
        "ambiguous_top_1_accuracy": top1_hits / len(ambiguous),
        "ambiguous_correct_in_context": in_context / len(ambiguous),
        "ambiguous_rescued_by_widening": (rescued / len(missed_top1)) if missed_top1 else 0.0,
        "ambiguous_avg_tools_in_context": mean(
            [float(o.tools_in_context) for _, o in ambiguous]
        ),
    }


def evaluate(
    dataset_path: str,
    router: ToolRouter,
    baseline: str,
    *,
    k: int = 5,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    write_files: bool = True,
) -> dict:
    """Evaluate one baseline over the dataset.

    Parameters
    ----------
    dataset_path:
        JSONL dataset produced by :mod:`toolrouter.bench.generate_dataset`.
    router:
        A configured :class:`~toolrouter.ToolRouter`. For the ``hybrid``
        baseline it must have been built with ``use_hybrid=True``.
    baseline:
        One of ``{"all_tools", "dense", "hybrid", "confidence_gate"}``.
    k:
        Retrieval budget. Also the gate's ``max_k``, so baselines are compared
        under an identical candidate budget.

    Returns
    -------
    dict
        ``metric_name -> value``, plus ``by_category`` and ``ambiguous_behaviour``
        breakdowns and a ``config`` block recording how the numbers were produced.
    """
    if baseline not in BASELINES:
        raise ValueError(f"Unknown baseline {baseline!r}. Expected one of {list(BASELINES)}.")

    dataset = load_dataset(dataset_path)

    # Warm the embedding cache before timing anything. The encoder is memoised
    # per query string, so whichever baseline runs first would otherwise absorb
    # the entire cold-start encoding cost and look ~50x slower than the others
    # (measured: 11.88 ms p50 vs 0.22 ms) -- an artefact of evaluation order,
    # not a real difference between the methods. Warming makes the reported
    # latencies measure search, which is what actually differs between them.
    for row in dataset:
        router.embedder.embed_text(row.query)

    rows: list[tuple[BenchQuery, BaselineOutcome]] = []
    for row in dataset:
        outcome = run_baseline(baseline, router, row.query, k=k)
        rows.append((row, outcome))

    results: dict = {
        "baseline": baseline,
        "k": k,
        **_aggregate(rows, baseline=baseline),
    }

    # Per-category breakdown -- accuracy by difficulty is far more credible
    # than a single blended number.
    categories = sorted({row.category for row, _ in rows})
    results["by_category"] = {
        category: _aggregate([(q, o) for q, o in rows if q.category == category], baseline=baseline)
        for category in categories
    }
    results["ambiguous_behaviour"] = _ambiguous_behaviour(rows)

    # Token reduction vs. the unrouted baseline, measured on the same corpus.
    # The reference is the *same* rendering the all_tools baseline is scored
    # with (header=False), so the unrouted baseline reports exactly 0.0%
    # reduction against itself rather than a spurious few percent caused by
    # comparing two slightly different renderings of the same tool list.
    all_tools_tokens = estimate_tokens(
        build_tool_prompt(router.registry.tools, style="json", header=False)
    )
    avg_tokens = results["avg_prompt_tokens"]
    results["all_tools_prompt_tokens"] = all_tools_tokens
    results["token_reduction_vs_all_tools"] = (
        (all_tools_tokens - avg_tokens) / all_tools_tokens if all_tools_tokens else 0.0
    )

    results["config"] = {
        "dataset": dataset_path,
        "dataset_queries": len(dataset),
        "router": router.stats(),
        "tokenizer": _tokenizer_name(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if write_files:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        result_path = target_dir / f"{baseline}.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        logger.info("Wrote %s", result_path)

        # Per-query predictions, for debugging and for walking through a single
        # ambiguous example live.
        detail_path = target_dir / f"{baseline}_predictions.jsonl"
        with detail_path.open("w", encoding="utf-8") as handle:
            for query_row, outcome in rows:
                handle.write(
                    json.dumps(
                        {
                            "query": query_row.query,
                            "correct_tool": query_row.correct_tool,
                            "category": query_row.category,
                            "predicted": outcome.ranked_tools,
                            "rank_of_correct": rank_of(
                                outcome.ranked_tools, query_row.correct_tool
                            ),
                            "scores": [round(s, 4) for s in outcome.scores],
                            "gate_mode": outcome.gate_mode,
                            "gate_widened": outcome.gate_widened,
                            "prompt_tokens": outcome.prompt_tokens,
                            "tools_in_context": outcome.tools_in_context,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return results


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #
_SUMMARY_HEADER = """# CommerceBench Results

Generated by `toolrouter.bench.evaluate` -- do not edit by hand.

"""


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_summary(all_results: dict[str, dict]) -> str:
    """Render the combined markdown summary for all evaluated baselines."""
    if not all_results:
        return _SUMMARY_HEADER + "_No results._\n"

    any_result = next(iter(all_results.values()))
    config = any_result.get("config", {})
    router_stats = config.get("router", {})

    lines = [_SUMMARY_HEADER.rstrip("\n"), ""]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Tools indexed:** {router_stats.get('tools', 'n/a')} "
                 f"across servers {router_stats.get('servers', [])}")
    lines.append(f"- **Embedding model:** `{router_stats.get('embedder', 'n/a')}` "
                 f"(dim {router_stats.get('embedder_dim', 'n/a')}"
                 f"{', FALLBACK MODE' if router_stats.get('embedder_is_fallback') else ''})")
    lines.append(f"- **Vector store:** `{router_stats.get('vector_store', 'n/a')}`")
    lines.append(f"- **BM25:** `{router_stats.get('bm25') or 'not built'}`")
    lines.append(f"- **Gate:** gap_threshold={router_stats.get('gap_threshold')}, "
                 f"score_floor={router_stats.get('score_floor')}")
    lines.append(f"- **Dataset:** `{config.get('dataset')}` "
                 f"({config.get('dataset_queries')} queries)")
    lines.append(f"- **Token counting:** `{config.get('tokenizer')}`")
    lines.append(f"- **Retrieval budget:** k={any_result.get('k')}")
    lines.append(f"- **Generated:** {config.get('generated_at')}")
    lines.append("")

    # -- headline table (this is what the README quotes) -- #
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "| Method | Top-1 Acc | Top-3 Acc | MRR | NDCG@5 | Avg Prompt Tokens | "
        "Token Reduction | Avg Tools in Context | p50 Latency | p95 Latency |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    labels = {
        "all_tools": "All tools (baseline)",
        "dense": "Dense only",
        "hybrid": "Dense + BM25",
        "confidence_gate": "Dense + confidence gate",
    }
    for baseline in BASELINES:
        result = all_results.get(baseline)
        if not result:
            continue
        lines.append(
            f"| {labels[baseline]} "
            f"| {_fmt_pct(result['top_1_accuracy'])} "
            f"| {_fmt_pct(result['top_3_accuracy'])} "
            f"| {result['mrr']:.3f} "
            f"| {result['ndcg_at_5']:.3f} "
            f"| {result['avg_prompt_tokens']:.0f} "
            f"| {_fmt_pct(result['token_reduction_vs_all_tools'])} "
            f"| {result['avg_tools_in_context']:.2f} "
            f"| {result['latency_p50_ms']:.2f} ms "
            f"| {result['latency_p95_ms']:.2f} ms |"
        )
    lines.append("")
    lines.append(
        "> The `all_tools` baseline has no ranking of its own -- its tool list is "
        "manifest order, so its Top-1/Top-3/MRR/NDCG columns measure manifest "
        "position, not relevance. Its meaningful column is prompt tokens. See "
        "`baselines.py` for the full explanation of this convention."
    )
    lines.append("")

    # -- recall table -- #
    lines.append("## Recall@k")
    lines.append("")
    lines.append("| Method | " + " | ".join(f"Recall@{k}" for k in RECALL_KS) + " |")
    lines.append("|---" * (len(RECALL_KS) + 1) + "|")
    for baseline in BASELINES:
        result = all_results.get(baseline)
        if not result:
            continue
        cells = " | ".join(_fmt_pct(result[f"recall_at_{k}"]) for k in RECALL_KS)
        lines.append(f"| {labels[baseline]} | {cells} |")
    lines.append("")
    lines.append(
        "> With exactly one relevant tool per query, Recall@k is by definition "
        "equal to Top-k accuracy. Both are reported because `BENCHMARK.md` asks "
        "for both; they are not independent signals."
    )
    lines.append("")

    # -- per-category accuracy -- #
    categories: list[str] = []
    for result in all_results.values():
        for category in result.get("by_category", {}):
            if category not in categories:
                categories.append(category)
    categories.sort()

    if categories:
        lines.append("## Top-1 accuracy by category")
        lines.append("")
        lines.append("| Method | " + " | ".join(f"{c} (n)" for c in categories) + " |")
        lines.append("|---" * (len(categories) + 1) + "|")
        for baseline in BASELINES:
            result = all_results.get(baseline)
            if not result:
                continue
            cells = []
            for category in categories:
                block = result.get("by_category", {}).get(category)
                if not block or not block.get("queries"):
                    cells.append("--")
                else:
                    cells.append(
                        f"{_fmt_pct(block['top_1_accuracy'])} ({block['queries']})"
                    )
            lines.append(f"| {labels[baseline]} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("## Top-3 accuracy by category")
        lines.append("")
        lines.append("| Method | " + " | ".join(categories) + " |")
        lines.append("|---" * (len(categories) + 1) + "|")
        for baseline in BASELINES:
            result = all_results.get(baseline)
            if not result:
                continue
            cells = []
            for category in categories:
                block = result.get("by_category", {}).get(category)
                cells.append(
                    _fmt_pct(block["top_3_accuracy"]) if block and block.get("queries") else "--"
                )
            lines.append(f"| {labels[baseline]} | " + " | ".join(cells) + " |")
        lines.append("")

    # -- ambiguous behaviour -- #
    lines.append("## Ambiguous-query behaviour")
    lines.append("")
    lines.append(
        "The metric most retrieval write-ups omit: when the router is genuinely "
        "unsure, does it widen the candidate set instead of confidently returning "
        "one wrong tool?"
    )
    lines.append("")
    lines.append(
        "| Method | Widen Rate | Top-1 Acc | Correct Tool in Context | "
        "Rescued by Widening | Avg Tools in Context |"
    )
    lines.append("|---|---|---|---|---|---|")
    for baseline in BASELINES:
        result = all_results.get(baseline)
        if not result:
            continue
        block = result.get("ambiguous_behaviour", {})
        if not block.get("ambiguous_queries"):
            continue
        lines.append(
            f"| {labels[baseline]} "
            f"| {_fmt_pct(block['ambiguous_widen_rate'])} "
            f"| {_fmt_pct(block['ambiguous_top_1_accuracy'])} "
            f"| {_fmt_pct(block['ambiguous_correct_in_context'])} "
            f"| {_fmt_pct(block['ambiguous_rescued_by_widening'])} "
            f"| {block['ambiguous_avg_tools_in_context']:.2f} |"
        )
    lines.append("")
    lines.append(
        "*Rescued by widening* = of the ambiguous queries where the top-ranked "
        "tool was wrong, the share where the correct tool was still in context "
        "for the LLM to choose. Fixed-top-k baselines cannot widen by "
        "construction, so their widen rate is 0%."
    )
    lines.append("")

    # -- gate mode distribution -- #
    gated = all_results.get("confidence_gate", {})
    if gated.get("gate_modes"):
        lines.append("## Confidence gate decisions")
        lines.append("")
        lines.append("| Mode | Queries | Share |")
        lines.append("|---|---|---|")
        total = sum(gated["gate_modes"].values())
        for mode, count in sorted(gated["gate_modes"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{mode}` | {count} | {_fmt_pct(count / total)} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def evaluate_all(
    dataset_path: str | None = None,
    router: ToolRouter | None = None,
    baselines: Sequence[str] | None = None,
    *,
    k: int = 5,
    manifest_path: str = "examples/swiggy_manifest.json",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, dict]:
    """Evaluate every baseline and write the combined ``summary.md``."""
    from .. import ToolRouter  # local import avoids a circular import at module load

    dataset_path = dataset_path or DEFAULT_DATASET
    baselines = list(baselines) if baselines else list(BASELINES)
    if router is None:
        router = ToolRouter.from_manifest(manifest_path, use_hybrid=True)
    elif "hybrid" in baselines and router.bm25 is None:
        logger.warning(
            "Router has no BM25 index; skipping the hybrid baseline. Rebuild with "
            "use_hybrid=True to include it."
        )
        baselines = [b for b in baselines if b != "hybrid"]

    all_results: dict[str, dict] = {}
    for baseline in baselines:
        logger.info("Evaluating baseline: %s", baseline)
        all_results[baseline] = evaluate(
            dataset_path, router, baseline, k=k, output_dir=output_dir
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.md"
    summary_path.write_text(render_summary(all_results), encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return all_results


def main() -> None:  # pragma: no cover - CLI
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run CommerceBench.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--manifest", default="examples/swiggy_manifest.json")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--baseline",
        action="append",
        choices=list(BASELINES),
        help="Evaluate only this baseline (repeatable). Default: all four.",
    )
    args = parser.parse_args()

    results = evaluate_all(
        dataset_path=args.dataset,
        baselines=args.baseline,
        k=args.k,
        manifest_path=args.manifest,
        output_dir=args.out,
    )
    print()
    print(render_summary(results))


if __name__ == "__main__":  # pragma: no cover
    main()
