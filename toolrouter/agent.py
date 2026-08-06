"""Agent-runtime wiring: routed tools -> LLM tool call -> MCP execution.

``BUILD_PLAN.md`` Weekend 2 asks for a real agent loop where the model sees
*only the routed subset*, not every tool in every connected server. This module
is that wiring, kept deliberately thin.

Two pieces:

* :func:`to_openai_tools` -- render routed :class:`~toolrouter.parser.manifest_parser.Tool`
  objects as OpenAI function-calling schemas. This is the format the OpenAI
  Agents SDK, LangGraph, and most other runtimes accept, so the same output
  drops into any of them.
* :class:`RoutedAgent` -- the loop itself: route the query, inject only the
  routed schemas, let the model choose, execute the chosen tool, return a
  :class:`AgentRun` recording every step.

Tool *execution* is pluggable via :class:`ToolExecutor`. The default
:class:`EchoToolExecutor` does not call a live MCP server -- it returns a
structured stub. That is a deliberate honesty constraint: this repository has no
verified live MCP endpoint, so pretending to call one would make the demo
misleading. Point :class:`RoutedAgent` at a real executor and the same loop
drives real tools unchanged.

Similarly, tool *selection* is pluggable via :class:`LLMClient`. With
``OPENAI_API_KEY`` set, :class:`OpenAIClient` asks a real model to choose. With
no key, :class:`HeuristicClient` picks the router's top-ranked candidate -- so
the loop is runnable and testable offline, and it reports which mode it used.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .parser.manifest_parser import Tool
from .router.prompt_builder import build_no_match_prompt, estimate_tokens

if TYPE_CHECKING:  # pragma: no cover
    from . import ToolRouter
    from .router.retrieve import RouteResult

__all__ = [
    "AgentRun",
    "AgentStep",
    "EchoToolExecutor",
    "HeuristicClient",
    "LLMClient",
    "OpenAIClient",
    "RoutedAgent",
    "ToolExecutor",
    "to_openai_tools",
    "SYSTEM_PROMPT",
]

logger = logging.getLogger(__name__)

#: Framing for the routed tool block. States the refusal contract explicitly:
#: a router that filters tools is only useful if the model declines when none of
#: the survivors fit, rather than forcing the closest one.
SYSTEM_PROMPT = (
    "You are an assistant that fulfils user requests by calling tools exposed by "
    "connected MCP servers.\n\n"
    "The tools below have been pre-selected by a semantic router as the ones most "
    "likely relevant to this specific request -- they are a subset, not the full "
    "catalogue. Choose the single best match and call it with the required "
    "parameters. If none of them genuinely fit the request, say so plainly instead "
    "of calling the closest available tool."
)


# --------------------------------------------------------------------------- #
# Schema rendering
# --------------------------------------------------------------------------- #
def to_openai_tools(tools: Sequence[Tool]) -> list[dict]:
    """Render tools as OpenAI function-calling schemas.

    The returned shape is the ``{"type": "function", "function": {...}}`` envelope
    used by the OpenAI Chat Completions and Agents APIs, and accepted (or
    trivially adaptable) by LangGraph, LiteLLM, and Anthropic's tool format.

    Examples
    --------
    >>> t = Tool("book_table", "Reserve a table.",
    ...          {"type": "object", "properties": {"party_size": {"type": "integer"}},
    ...           "required": ["party_size"]}, "dineout")
    >>> schema = to_openai_tools([t])[0]
    >>> schema["function"]["name"]
    'book_table'
    >>> schema["function"]["parameters"]["required"]
    ['party_size']
    """
    rendered: list[dict] = []
    for tool in tools:
        parameters = dict(tool.parameters) if isinstance(tool.parameters, dict) else {}
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        rendered.append(
            {
                "type": "function",
                "function": {
                    # Server is folded into the description rather than the name:
                    # tool names must stay exactly as the MCP server declares them
                    # or the resulting call is not dispatchable.
                    "name": tool.name,
                    "description": (
                        f"[{tool.server}] {tool.description}"
                        if tool.server
                        else tool.description
                    ),
                    "parameters": parameters,
                },
            }
        )
    return rendered


# --------------------------------------------------------------------------- #
# Pluggable seams
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    """Chooses one tool (or none) from the routed subset."""

    name: str

    def choose_tool(
        self, query: str, tools: Sequence[Tool], *, system_prompt: str
    ) -> tuple[str | None, dict, str]:
        """Return ``(tool_name_or_None, arguments, rationale)``."""


class ToolExecutor(Protocol):
    """Executes a chosen tool against a real (or simulated) MCP server."""

    name: str

    def execute(self, tool: Tool, arguments: dict) -> dict:
        """Return the tool's result payload."""


class HeuristicClient:
    """Offline stand-in for an LLM: takes the router's top-ranked candidate.

    This is *not* a model, and the distinction matters for interpreting a run:
    it measures the router in isolation (does the top candidate answer the
    query?) with no reasoning step layered on top. :attr:`name` says so, and
    :class:`AgentRun` records it.
    """

    name = "heuristic:top-ranked-candidate"

    def choose_tool(
        self, query: str, tools: Sequence[Tool], *, system_prompt: str
    ) -> tuple[str | None, dict, str]:
        if not tools:
            return None, {}, "No tools survived the confidence gate; nothing to choose."
        chosen = tools[0]
        # Required parameters are left as explicit placeholders rather than
        # invented values -- fabricating a restaurant_id would make the demo
        # look like it worked when it only guessed.
        arguments = {name: f"<{name}>" for name in chosen.required_parameters}
        return (
            chosen.name,
            arguments,
            f"No LLM configured; selected the router's top-ranked candidate "
            f"{chosen.name!r} out of {len(tools)} routed tool(s).",
        )


class OpenAIClient:
    """Real tool selection via the OpenAI Chat Completions API.

    Requires the ``openai`` package and ``OPENAI_API_KEY``. Raises on
    construction if either is missing, so callers fail fast and fall back
    explicitly rather than silently degrading.
    """

    def __init__(self, model: str = "gpt-4o-mini", *, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAIClient.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'openai' package is not installed. `pip install openai` or use "
                "HeuristicClient."
            ) from exc
        self._client = OpenAI(api_key=key)
        self.model = model
        self.name = f"openai:{model}"

    def choose_tool(
        self, query: str, tools: Sequence[Tool], *, system_prompt: str
    ) -> tuple[str | None, dict, str]:
        if not tools:
            return None, {}, "No tools survived the confidence gate; nothing to choose."

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            tools=to_openai_tools(tools),
            tool_choice="auto",
        )
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", None)
        if not calls:
            return (
                None,
                {},
                f"Model declined to call any routed tool: {message.content or '(no text)'}",
            )
        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": call.function.arguments}
        return (
            call.function.name,
            arguments,
            f"Model selected {call.function.name!r} from {len(tools)} routed tool(s).",
        )


class EchoToolExecutor:
    """Simulated MCP execution -- echoes the call back as a structured stub.

    Deliberately does not fake plausible-looking business data. A stub that
    returned ``{"reservation_id": "R-4823", "status": "confirmed"}`` would make
    this demo indistinguishable from one wired to a live server. Swap in a real
    executor (an MCP client, an HTTP call) and the agent loop is unchanged.
    """

    name = "echo:simulated"

    def execute(self, tool: Tool, arguments: dict) -> dict:
        missing = [p for p in tool.required_parameters if p not in arguments]
        return {
            "_simulated": True,
            "_note": (
                "No live MCP server is connected. This is a structured stub "
                "confirming which tool the agent selected and with what arguments."
            ),
            "tool": tool.name,
            "server": tool.server,
            "arguments": arguments,
            "missing_required_parameters": missing,
        }


# --------------------------------------------------------------------------- #
# Run records
# --------------------------------------------------------------------------- #
@dataclass
class AgentStep:
    """One stage of an agent run, with its timing."""

    stage: str
    detail: str
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass
class AgentRun:
    """The full record of one query through the routed agent loop."""

    query: str
    routed_tools: list[str]
    gate_mode: str
    chosen_tool: str | None
    arguments: dict = field(default_factory=dict)
    result: dict | None = None
    rationale: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    prompt_tokens_routed: int = 0
    prompt_tokens_unrouted: int = 0
    llm_backend: str = ""
    executor_backend: str = ""
    latency_ms: float = 0.0

    @property
    def token_reduction(self) -> float:
        """Fraction of tool-schema tokens saved versus injecting every tool."""
        if not self.prompt_tokens_unrouted:
            return 0.0
        saved = self.prompt_tokens_unrouted - self.prompt_tokens_routed
        return saved / self.prompt_tokens_unrouted

    @property
    def called_a_tool(self) -> bool:
        return self.chosen_tool is not None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "routed_tools": list(self.routed_tools),
            "gate_mode": self.gate_mode,
            "chosen_tool": self.chosen_tool,
            "arguments": self.arguments,
            "result": self.result,
            "rationale": self.rationale,
            "steps": [s.to_dict() for s in self.steps],
            "prompt_tokens_routed": self.prompt_tokens_routed,
            "prompt_tokens_unrouted": self.prompt_tokens_unrouted,
            "token_reduction": round(self.token_reduction, 6),
            "llm_backend": self.llm_backend,
            "executor_backend": self.executor_backend,
            "latency_ms": round(self.latency_ms, 3),
        }


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
class RoutedAgent:
    """An agent that only ever sees the routed tool subset.

    Parameters
    ----------
    router:
        A configured :class:`~toolrouter.ToolRouter`.
    llm:
        Tool-selection client. Defaults to :class:`OpenAIClient` when
        ``OPENAI_API_KEY`` is present, else :class:`HeuristicClient`.
    executor:
        Tool execution backend. Defaults to :class:`EchoToolExecutor`.
    system_prompt:
        Framing prepended to the routed tool block.

    Examples
    --------
    The tool below is what this actually returns on the sample manifest under the
    default offline embedder, not an idealised answer.

    >>> from toolrouter import ToolRouter                        # doctest: +SKIP
    >>> agent = RoutedAgent(ToolRouter.from_manifest("examples/swiggy_manifest.json"))
    ...                                                          # doctest: +SKIP
    >>> run = agent.run("book a table for four tonight")         # doctest: +SKIP
    >>> run.chosen_tool                                          # doctest: +SKIP
    'book_restaurant_table'
    """

    def __init__(
        self,
        router: ToolRouter,
        *,
        llm: LLMClient | None = None,
        executor: ToolExecutor | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.router = router
        self.llm: LLMClient = llm if llm is not None else _default_llm()
        self.executor: ToolExecutor = (
            executor if executor is not None else EchoToolExecutor()
        )
        self.system_prompt = system_prompt

    # -- main entry point --------------------------------------------------- #
    def run(self, query: str, **route_kwargs: Any) -> AgentRun:
        """Route, select, execute. Returns the full :class:`AgentRun` record."""
        overall_start = time.perf_counter()
        steps: list[AgentStep] = []

        # 1. route ---------------------------------------------------------- #
        result: RouteResult = self.router.route(query, **route_kwargs)
        steps.append(
            AgentStep(
                stage="route",
                detail=(
                    f"gate={result.gate.get('mode')}, "
                    f"routed {len(result.tools)}/{len(self.router.registry)} tools: "
                    f"{result.tool_names or '[]'}"
                ),
                duration_ms=result.latency_ms,
            )
        )

        # 2. build the prompt the model will actually see -------------------- #
        routed_prompt = (
            f"{self.system_prompt}\n\n{self.router.build_prompt(result, header=False)}"
            if result.tools
            else build_no_match_prompt()
        )
        tokens_routed = estimate_tokens(routed_prompt)
        tokens_unrouted = estimate_tokens(self.router.all_tools_prompt())
        steps.append(
            AgentStep(
                stage="build_prompt",
                detail=(
                    f"{tokens_routed} tool-schema tokens injected vs "
                    f"{tokens_unrouted} unrouted"
                ),
            )
        )

        # 3. the gate can end the run before any model is involved ----------- #
        if not result.tools:
            steps.append(
                AgentStep(
                    stage="refuse",
                    detail=(
                        "Confidence gate reported no confident match; the agent "
                        "declines rather than calling an unrelated tool."
                    ),
                )
            )
            return AgentRun(
                query=query,
                routed_tools=[],
                gate_mode=str(result.gate.get("mode")),
                chosen_tool=None,
                rationale=str(result.gate.get("reason", "")),
                steps=steps,
                prompt_tokens_routed=tokens_routed,
                prompt_tokens_unrouted=tokens_unrouted,
                llm_backend=self.llm.name,
                executor_backend=self.executor.name,
                latency_ms=(time.perf_counter() - overall_start) * 1000.0,
            )

        # 4. selection ------------------------------------------------------ #
        select_start = time.perf_counter()
        chosen_name, arguments, rationale = self.llm.choose_tool(
            query, result.tools, system_prompt=self.system_prompt
        )
        steps.append(
            AgentStep(
                stage="select",
                detail=f"[{self.llm.name}] {rationale}",
                duration_ms=(time.perf_counter() - select_start) * 1000.0,
            )
        )

        if chosen_name is None:
            return AgentRun(
                query=query,
                routed_tools=result.tool_names,
                gate_mode=str(result.gate.get("mode")),
                chosen_tool=None,
                rationale=rationale,
                steps=steps,
                prompt_tokens_routed=tokens_routed,
                prompt_tokens_unrouted=tokens_unrouted,
                llm_backend=self.llm.name,
                executor_backend=self.executor.name,
                latency_ms=(time.perf_counter() - overall_start) * 1000.0,
            )

        # A model can hallucinate a name that was never in the routed subset.
        # Surface that loudly instead of dispatching it.
        tool = self.router.registry.by_name(chosen_name)
        if tool is None or chosen_name not in result.tool_names:
            steps.append(
                AgentStep(
                    stage="reject",
                    detail=(
                        f"Selected tool {chosen_name!r} was not in the routed subset "
                        f"{result.tool_names}; refusing to dispatch it."
                    ),
                )
            )
            return AgentRun(
                query=query,
                routed_tools=result.tool_names,
                gate_mode=str(result.gate.get("mode")),
                chosen_tool=None,
                rationale=f"Rejected out-of-subset selection {chosen_name!r}.",
                steps=steps,
                prompt_tokens_routed=tokens_routed,
                prompt_tokens_unrouted=tokens_unrouted,
                llm_backend=self.llm.name,
                executor_backend=self.executor.name,
                latency_ms=(time.perf_counter() - overall_start) * 1000.0,
            )

        # 5. execute -------------------------------------------------------- #
        execute_start = time.perf_counter()
        payload = self.executor.execute(tool, arguments)
        steps.append(
            AgentStep(
                stage="execute",
                detail=f"[{self.executor.name}] called {tool.name} on '{tool.server}'",
                duration_ms=(time.perf_counter() - execute_start) * 1000.0,
            )
        )

        return AgentRun(
            query=query,
            routed_tools=result.tool_names,
            gate_mode=str(result.gate.get("mode")),
            chosen_tool=tool.name,
            arguments=arguments,
            result=payload,
            rationale=rationale,
            steps=steps,
            prompt_tokens_routed=tokens_routed,
            prompt_tokens_unrouted=tokens_unrouted,
            llm_backend=self.llm.name,
            executor_backend=self.executor.name,
            latency_ms=(time.perf_counter() - overall_start) * 1000.0,
        )

    def __repr__(self) -> str:
        return (
            f"RoutedAgent(tools={len(self.router.registry)}, llm={self.llm.name!r}, "
            f"executor={self.executor.name!r})"
        )


def _default_llm() -> LLMClient:
    """OpenAI when a key is available, the offline heuristic otherwise."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to HeuristicClient: %s", exc)
    return HeuristicClient()
