#!/usr/bin/env python3
"""Agent demo: a real tool-call loop that only ever sees the routed subset.

This is the ``BUILD_PLAN.md`` Weekend 2 deliverable -- "wire into an agent
runtime so a real query -> routed tools -> actual tool call loop works". It
shows the whole chain, and crucially it shows the chain *refusing* to call
anything when the confidence gate finds no confident match.

Two honesty notes, both surfaced in the output rather than buried here:

* **Tool selection.** With ``OPENAI_API_KEY`` set, a real model chooses from the
  routed subset. Without it, the offline heuristic takes the router's top-ranked
  candidate -- which measures the router alone, with no reasoning layer.
* **Tool execution.** No verified live MCP endpoint ships with this repo, so
  execution is a structured stub that echoes the call. Pass a real
  ``ToolExecutor`` and the same loop drives real tools unchanged.

Usage
-----
    python examples/agent_demo.py
    python examples/agent_demo.py --hybrid
    python examples/agent_demo.py --query "cancel my reservation"
    python examples/agent_demo.py --json          # machine-readable run records
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolrouter import ToolRouter  # noqa: E402
from toolrouter.agent import RoutedAgent, to_openai_tools  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "examples" / "swiggy_manifest.json"

DEMO_QUERIES: list[str] = [
    "reserve a table for 4 at 8pm tonight",
    "add two butter naan to my order",
    "is my grocery delivery here yet",
    "what is the capital of peru",  # must refuse -- nothing serves this
]

RULE = "=" * 78
THIN = "-" * 78


def _print_run(agent: RoutedAgent, query: str, *, show_schema: bool) -> None:
    run = agent.run(query)

    print(RULE)
    print(f'QUERY: "{query}"')
    print(RULE)

    for step in run.steps:
        timing = f"  ({step.duration_ms:.2f} ms)" if step.duration_ms else ""
        print(f"  [{step.stage}]{timing}")
        print(f"      {step.detail}")

    print()
    if run.called_a_tool:
        print(f"  TOOL CALLED : {run.chosen_tool}")
        print(f"  ARGUMENTS   : {json.dumps(run.arguments)}")
        print(f"  RESULT      : {json.dumps(run.result)}")
    else:
        print("  TOOL CALLED : none")
        print(f"  WHY         : {run.rationale}")

    print(
        f"  TOKENS      : {run.prompt_tokens_routed} routed vs "
        f"{run.prompt_tokens_unrouted} unrouted "
        f"({run.token_reduction * 100:.1f}% smaller)"
    )
    print(f"  TOTAL       : {run.latency_ms:.2f} ms")

    if show_schema and run.routed_tools:
        tools = [agent.router.registry.require(n) for n in run.routed_tools]
        print("\n  OpenAI function schemas handed to the model:")
        print(THIN)
        print(json.dumps(to_openai_tools(tools), indent=2))
        print(THIN)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the routed agent loop over the mock manifest."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--hybrid", action="store_true", help="Enable BM25 fusion.")
    parser.add_argument(
        "--offline", action="store_true", help="Force the hash embedder."
    )
    parser.add_argument(
        "--query", action="append", help="Run this query instead of the demo set."
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Print the OpenAI function schemas sent to the model.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit run records as JSON only."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.offline:
        os.environ["TOOLROUTER_FORCE_FALLBACK"] = "1"

    router = ToolRouter.from_manifest(args.manifest, use_hybrid=args.hybrid)
    agent = RoutedAgent(router)
    queries = args.query or DEMO_QUERIES

    if args.json:
        print(json.dumps([agent.run(q).to_dict() for q in queries], indent=2))
        return 0

    stats = router.stats()
    print(RULE)
    print("toolrouter agent demo")
    print(RULE)
    print(f"tools available : {stats['tools']} across {stats['servers']}")
    print(f"embedder        : {stats['embedder']}")
    print(f"tool selection  : {agent.llm.name}")
    if agent.llm.name.startswith("heuristic"):
        print("                  ^ no OPENAI_API_KEY set. This takes the router's")
        print("                    top candidate; it is not model reasoning.")
    print(f"tool execution  : {agent.executor.name}")
    print("                  ^ simulated. No live MCP server is connected;")
    print("                    results are structured stubs, not real data.")
    print()

    for query in queries:
        _print_run(agent, query, show_schema=args.show_schema)

    print(RULE)
    print("The model never saw the full tool catalogue -- only the routed subset,")
    print("and nothing at all when the gate found no confident match.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
