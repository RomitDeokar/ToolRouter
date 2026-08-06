"""Unified command-line interface.

One entry point over the whole pipeline, so the library is usable without
writing Python::

    toolrouter route "book a table for four tonight"
    toolrouter tools --server dineout
    toolrouter prompt "order paneer" --style json
    toolrouter agent "cancel my reservation"
    toolrouter bench --k 5
    toolrouter calibrate
    toolrouter dataset --per-tool 12

The per-module ``python -m toolrouter.bench.*`` entry points still work; the
``bench``, ``calibrate`` and ``dataset`` subcommands delegate to them so there is
exactly one implementation of each behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

from . import ToolRouter, __version__

__all__ = ["main", "build_parser"]

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = "examples/swiggy_manifest.json"


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #
def _add_router_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand that builds a router."""
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        help=(
            "Manifest path or URL. Repeat to route across several MCP servers at "
            f"once (default: {DEFAULT_MANIFEST})."
        ),
    )
    parser.add_argument(
        "--hybrid", action="store_true", help="Fuse BM25 lexical scores with dense."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the deterministic hash embedder (no model download).",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=None,
        help="Override the confidence gate's top1-top2 gap threshold.",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=None,
        help="Override the gate's absolute score floor. 0 disables it.",
    )
    parser.add_argument("--min-k", type=int, default=None)
    parser.add_argument("--max-k", type=int, default=None)


def _build_router(args: argparse.Namespace) -> ToolRouter:
    if getattr(args, "offline", False):
        os.environ["TOOLROUTER_FORCE_FALLBACK"] = "1"

    manifests: list[str] = args.manifest or [DEFAULT_MANIFEST]
    overrides = {
        key: value
        for key, value in (
            ("gap_threshold", args.gap_threshold),
            ("score_floor", args.score_floor),
            ("min_k", args.min_k),
            ("max_k", args.max_k),
        )
        if value is not None
    }

    if len(manifests) == 1:
        return ToolRouter.from_manifest(
            manifests[0], use_hybrid=args.hybrid, **overrides
        )
    return ToolRouter.from_manifests(manifests, use_hybrid=args.hybrid, **overrides)


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# route
# --------------------------------------------------------------------------- #
def _cmd_route(args: argparse.Namespace) -> int:
    router = _build_router(args)
    result = router.route(args.query, k=args.k, server=args.server)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f'query : "{result.query}"')
    print(
        f"router: {len(router.registry)} tools, {router.embedder.backend}"
        f"{' [FALLBACK]' if router.embedder.is_fallback else ''}"
    )
    print()
    print(f"gate  : {result.gate.get('mode')} -- {result.gate.get('reason')}")
    print()

    if not result.candidates:
        print("no candidates retrieved.")
        return 0

    print("candidates (* = kept by the gate):")
    kept = set(result.tool_names)
    for candidate in result.candidates:
        marker = "*" if candidate.tool.name in kept else " "
        print(
            f"  {marker} {candidate.score:.4f}  {candidate.tool.name:<30} "
            f"({candidate.tool.server})"
        )

    if args.explain:
        print("\nexplanations:")
        for row in result.explanation:
            print(f"  #{row['rank']} {row['tool'] or '(none)'}: {row['reason']}")

    print(f"\nlatency: {result.latency_ms:.2f} ms")
    # Non-zero exit when nothing matched, so shell pipelines can branch on it.
    return 0 if result.tools else 2


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
def _cmd_prompt(args: argparse.Namespace) -> int:
    router = _build_router(args)
    if args.all:
        print(router.all_tools_prompt(style=args.style))
        return 0
    result = router.route(args.query, k=args.k)
    print(router.build_prompt(result, style=args.style))
    return 0 if result.tools else 2


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def _cmd_tools(args: argparse.Namespace) -> int:
    router = _build_router(args)
    tools = (
        router.registry.by_server(args.server) if args.server else router.registry.tools
    )

    if args.json:
        print(json.dumps([t.to_dict() for t in tools], indent=2))
        return 0

    if not tools:
        print(f"no tools found{f' for server {args.server!r}' if args.server else ''}.")
        return 2

    width = max(len(t.name) for t in tools)
    print(f"{len(tools)} tool(s) across servers {router.registry.servers}:\n")
    for tool in tools:
        required = ", ".join(tool.required_parameters) or "-"
        print(f"  {tool.name:<{width}}  [{tool.server}]")
        print(f"  {'':<{width}}  {tool.description}")
        print(f"  {'':<{width}}  required: {required}")
    return 0


# --------------------------------------------------------------------------- #
# agent
# --------------------------------------------------------------------------- #
def _cmd_agent(args: argparse.Namespace) -> int:
    from .agent import RoutedAgent

    router = _build_router(args)
    agent = RoutedAgent(router)
    run = agent.run(args.query)

    if args.json:
        print(json.dumps(run.to_dict(), indent=2))
        return 0 if run.called_a_tool else 2

    print(f'query : "{run.query}"')
    print(f"llm   : {run.llm_backend}")
    print(f"exec  : {run.executor_backend}")
    print()
    for step in run.steps:
        timing = f" ({step.duration_ms:.2f} ms)" if step.duration_ms else ""
        print(f"  [{step.stage}]{timing} {step.detail}")
    print()
    if run.called_a_tool:
        print(f"called   : {run.chosen_tool}({json.dumps(run.arguments)})")
        print(f"result   : {json.dumps(run.result)}")
    else:
        print("called   : none")
        print(f"why      : {run.rationale}")
    print(
        f"tokens   : {run.prompt_tokens_routed} routed vs "
        f"{run.prompt_tokens_unrouted} unrouted "
        f"({_fmt_pct(run.token_reduction)} smaller)"
    )
    return 0 if run.called_a_tool else 2


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def _cmd_stats(args: argparse.Namespace) -> int:
    router = _build_router(args)
    stats = router.stats()
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    for key, value in stats.items():
        print(f"{key:<22} {value}")
    if stats["embedder_is_fallback"]:
        print(
            "\nWARNING: the hash fallback embedder is active. Retrieval is lexical, "
            "not semantic -- do not report benchmark numbers from this mode."
        )
    return 0


# --------------------------------------------------------------------------- #
# bench / calibrate / dataset -- delegate to the bench modules
# --------------------------------------------------------------------------- #
def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench.evaluate import evaluate_all, render_summary

    if args.offline:
        os.environ["TOOLROUTER_FORCE_FALLBACK"] = "1"
    manifests = args.manifest or [DEFAULT_MANIFEST]
    router = (
        ToolRouter.from_manifest(manifests[0], use_hybrid=True)
        if len(manifests) == 1
        else ToolRouter.from_manifests(manifests, use_hybrid=True)
    )
    results = evaluate_all(
        dataset_path=args.dataset,
        router=router,
        baselines=args.baseline,
        k=args.k,
        output_dir=args.out,
    )
    print()
    print(render_summary(results))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .bench.calibrate import main as calibrate_main

    argv = ["--dataset", args.dataset, "--out", args.out, "--k", str(args.k)]
    for manifest in args.manifest or [DEFAULT_MANIFEST]:
        argv += ["--manifest", manifest]
    return _delegate(calibrate_main, argv)


def _cmd_dataset(args: argparse.Namespace) -> int:
    from .bench.generate_dataset import main as dataset_main

    argv = [
        "--out", args.out,
        "--per-tool", str(args.per_tool),
        "--seed", str(args.seed),
        "--manifest", (args.manifest or [DEFAULT_MANIFEST])[0],
    ]
    if args.no_llm:
        argv.append("--no-llm")
    return _delegate(dataset_main, argv)


def _delegate(entry: object, argv: Sequence[str]) -> int:
    """Call a module ``main()`` with a synthetic ``sys.argv``."""
    saved = sys.argv
    sys.argv = ["toolrouter", *argv]
    try:
        entry()  # type: ignore[operator]
    except SystemExit as exc:  # a delegated main() may exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolrouter",
        description="Semantic tool retrieval for MCP agents.",
    )
    parser.add_argument("--version", action="version", version=f"toolrouter {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show library logs."
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # route
    route = sub.add_parser("route", help="Route a query and show the gate's decision.")
    route.add_argument("query")
    route.add_argument("--k", type=int, default=None, help="Candidate pool size.")
    route.add_argument("--server", default=None, help="Restrict to one MCP server.")
    route.add_argument("--explain", action="store_true", help="Print per-tool reasons.")
    route.add_argument("--json", action="store_true")
    _add_router_flags(route)
    route.set_defaults(func=_cmd_route)

    # prompt
    prompt = sub.add_parser("prompt", help="Print the LLM-ready tool prompt block.")
    prompt.add_argument("query", nargs="?", default="")
    prompt.add_argument("--k", type=int, default=None)
    prompt.add_argument("--style", choices=["compact", "json"], default="compact")
    prompt.add_argument(
        "--all", action="store_true", help="Render every tool (the unrouted baseline)."
    )
    prompt.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    _add_router_flags(prompt)
    prompt.set_defaults(func=_cmd_prompt)

    # tools
    tools = sub.add_parser("tools", help="List the indexed tools.")
    tools.add_argument("--server", default=None)
    tools.add_argument("--json", action="store_true")
    _add_router_flags(tools)
    tools.set_defaults(func=_cmd_tools)

    # agent
    agent = sub.add_parser("agent", help="Run the routed agent loop on one query.")
    agent.add_argument("query")
    agent.add_argument("--json", action="store_true")
    _add_router_flags(agent)
    agent.set_defaults(func=_cmd_agent)

    # stats
    stats = sub.add_parser("stats", help="Show the router's configuration.")
    stats.add_argument("--json", action="store_true")
    _add_router_flags(stats)
    stats.set_defaults(func=_cmd_stats)

    # bench
    bench = sub.add_parser("bench", help="Run CommerceBench over all four baselines.")
    bench.add_argument("--dataset", default="toolrouter/bench/dataset.jsonl")
    bench.add_argument("--out", default="bench_results")
    bench.add_argument("--k", type=int, default=5)
    bench.add_argument(
        "--baseline",
        action="append",
        choices=["all_tools", "dense", "hybrid", "confidence_gate"],
        help="Evaluate only this baseline (repeatable).",
    )
    _add_router_flags(bench)
    bench.set_defaults(func=_cmd_bench)

    # calibrate
    calibrate = sub.add_parser("calibrate", help="Sweep the gate's thresholds.")
    calibrate.add_argument("--dataset", default="toolrouter/bench/dataset.jsonl")
    calibrate.add_argument("--out", default="bench_results/calibration.md")
    calibrate.add_argument("--k", type=int, default=5)
    _add_router_flags(calibrate)
    calibrate.set_defaults(func=_cmd_calibrate)

    # dataset
    dataset = sub.add_parser("dataset", help="Regenerate the CommerceBench dataset.")
    dataset.add_argument("--out", default="toolrouter/bench/dataset.jsonl")
    dataset.add_argument("--per-tool", type=int, default=12)
    dataset.add_argument("--seed", type=int, default=20240501)
    dataset.add_argument("--no-llm", action="store_true", help="Force template mode.")
    _add_router_flags(dataset)
    dataset.set_defaults(func=_cmd_dataset)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
