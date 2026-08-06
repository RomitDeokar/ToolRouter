#!/usr/bin/env python3
"""End-to-end quickstart: manifest -> routed tools -> LLM-ready prompt.

Runs the full pipeline over the mock multi-server manifest and prints, for each
query: the routed tools with scores, the gate's decision and *why*, the
per-candidate explanations, and the exact prompt block an agent would receive.

The query set is chosen to exercise every branch of the confidence gate rather
than to look good in a screenshot:

* a **clean** query, where one tool obviously wins -> gate narrows to top-1;
* an **ambiguous** query, where sibling tools across two servers are both
  plausible -> gate widens k;
* a **typo/informal** query, to show lexical robustness;
* an **out-of-domain** query, which no tool can serve -> gate returns
  ``no_confident_match`` instead of forcing a guess.

Usage
-----
    python examples/quickstart.py                    # dense retrieval
    python examples/quickstart.py --hybrid           # dense + BM25 fusion
    python examples/quickstart.py --offline          # force the hash embedder
    python examples/quickstart.py --query "your own query here"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running this file directly from a clone without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolrouter import ToolRouter  # noqa: E402
from toolrouter.router.prompt_builder import estimate_tokens  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "examples" / "swiggy_manifest.json"

#: (query, why this query is in the set) -- one per gate branch.
DEMO_QUERIES: list[tuple[str, str]] = [
    (
        "book a table for four tonight at 8pm",
        "clean -- one tool obviously wins, the gate should narrow to top-1",
    ),
    (
        "order paneer",
        "ambiguous -- groceries to cook (instamart) or a ready meal (food)?",
    ),
    (
        "wheres my delivry guy rn",
        "typo/informal -- no exact tool tokens, tests robustness",
    ),
    (
        "what is the weather in tokyo tomorrow",
        "out-of-domain -- no tool serves this; the gate must refuse to guess",
    ),
]

RULE = "=" * 78
THIN = "-" * 78


def _print_header(router: ToolRouter, manifest: Path) -> None:
    stats = router.stats()
    print(RULE)
    print("toolrouter quickstart")
    print(RULE)
    print(f"manifest      : {manifest}")
    print(f"tools indexed : {stats['tools']} across servers {stats['servers']}")
    print(f"embedder      : {stats['embedder']} (dim {stats['embedder_dim']})")
    if stats["embedder_is_fallback"]:
        print("                ^ HASH FALLBACK -- lexical, not semantic. Results")
        print("                  from this mode are not benchmark-quality.")
    print(f"vector store  : {stats['vector_store']}")
    print(f"bm25          : {stats['bm25'] or 'not built (dense only)'}")
    print(f"gate          : gap_threshold={stats['gap_threshold']}, "
          f"score_floor={stats['score_floor']}, "
          f"min_k={stats['min_k']}, max_k={stats['max_k']}")
    print()


def _print_query(router: ToolRouter, query: str, note: str, *, hybrid: bool) -> None:
    result = router.route(query, hybrid=hybrid)

    print(RULE)
    print(f'QUERY: "{query}"')
    if note:
        print(f"       ({note})")
    print(RULE)

    # -- 1. candidate pool, before the gate ------------------------------- #
    print("\n[1] retrieved candidates (pre-gate)")
    if not result.candidates:
        print("    <none>")
    for candidate in result.candidates:
        selected = "*" if candidate.tool.name in result.tool_names else " "
        detail = ""
        if candidate.source == "hybrid":
            detail = (
                f"   [dense {candidate.components.get('dense', 0):.3f}"
                f" / bm25 {candidate.components.get('bm25', 0):.3f}]"
            )
        print(
            f"  {selected} {candidate.score:.4f}  "
            f"{candidate.tool.name:<28} ({candidate.tool.server}){detail}"
        )
    print("    (* = kept by the confidence gate)")

    # -- 2. the gate's decision ------------------------------------------- #
    gate = result.gate
    print(f"\n[2] confidence gate -> {gate.get('mode')}")
    gap = gate.get("gap")
    print(f"    top score  : {gate.get('top_score')}")
    print(f"    runner-up  : {gate.get('runner_up_score')}")
    print(f"    gap        : {'n/a' if gap is None else f'{gap:.4f}'} "
          f"(threshold {gate.get('gap_threshold')})")
    print(f"    selected k : {gate.get('selected_k')}")
    print(f"    reason     : {gate.get('reason')}")

    # -- 3. routed tools -------------------------------------------------- #
    print("\n[3] routed tools")
    if result.tools:
        for name in result.tool_names:
            print(f"    - {name}")
    else:
        print("    <none -- no confident match>")

    # -- 4. explanations -------------------------------------------------- #
    print("\n[4] explanations")
    for row in result.explanation:
        print(f"    #{row['rank']} {row['tool'] or '(no tool)'}")
        print(f"       {row['reason']}")

    # -- 5. the prompt the LLM actually sees ------------------------------ #
    prompt = router.build_prompt(result)
    routed_tokens = estimate_tokens(prompt)
    all_tokens = estimate_tokens(router.all_tools_prompt())
    saving = (1 - routed_tokens / all_tokens) if all_tokens else 0.0

    print("\n[5] prompt block injected into the agent")
    print(THIN)
    print(prompt)
    print(THIN)
    print(
        f"    {routed_tokens} tokens routed vs {all_tokens} tokens unrouted "
        f"-> {saving * 100:.1f}% smaller"
    )
    print(f"    retrieval latency: {result.latency_ms:.2f} ms")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the toolrouter pipeline end to end and print every stage."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Manifest path or URL (default: the bundled mock manifest).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Fuse BM25 lexical scores with dense scores.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the deterministic hash embedder (no model download).",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Run this query instead of the built-in demo set (repeatable).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show library logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.offline:
        os.environ["TOOLROUTER_FORCE_FALLBACK"] = "1"

    manifest = Path(args.manifest)
    if not str(args.manifest).startswith(("http://", "https://")) and not manifest.is_file():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 1

    router = ToolRouter.from_manifest(args.manifest, use_hybrid=args.hybrid)
    _print_header(router, manifest)

    queries = (
        [(q, "") for q in args.query] if args.query else DEMO_QUERIES
    )
    for query, note in queries:
        _print_query(router, query, note, hybrid=args.hybrid)

    print(RULE)
    print("Next: `python -m toolrouter.bench.evaluate` for the full benchmark,")
    print("      `python -m toolrouter.bench.calibrate` to re-tune the gate.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
