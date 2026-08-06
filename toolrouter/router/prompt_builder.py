"""Prompt builder: routed tools -> minimal schema block for the LLM.

This is where the token saving is actually realised. The agent's system prompt
receives only the routed subset in a compact form -- name, description, and a
flattened parameter list -- instead of the full JSON Schema for every tool in
every connected MCP server.

``build_tool_prompt`` is the contract from ``ARCHITECTURE.md``. The ``style``
parameter exists because the benchmark needs to measure token counts for both
this compact rendering *and* the raw-JSON rendering that unrouted agents use.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from ..parser.manifest_parser import Tool

__all__ = [
    "build_tool_prompt",
    "build_no_match_prompt",
    "estimate_tokens",
    "render_tool_compact",
    "render_tool_json",
]


def _type_of(schema: dict) -> str:
    """Best-effort JSON Schema type label, including ``enum`` and array items."""
    if not isinstance(schema, dict):
        return "any"
    if "enum" in schema and isinstance(schema["enum"], Sequence):
        options = "|".join(str(o) for o in schema["enum"])
        return f"enum({options})"
    type_name = schema.get("type")
    if isinstance(type_name, list):
        type_name = "|".join(str(t) for t in type_name)
    if type_name == "array":
        items = schema.get("items")
        inner = _type_of(items) if isinstance(items, dict) else "any"
        return f"array<{inner}>"
    return str(type_name) if type_name else "any"


def render_tool_compact(tool: Tool, *, include_examples: bool = False) -> str:
    """Render one tool as a compact, LLM-readable block.

    Example output::

        - book_table(dineout)
          Reserve a table at a restaurant for a given time and party size.
          params: restaurant_id: string (required), party_size: integer (required),
                  time: string, seating: enum(indoor|outdoor)
    """
    lines = [f"- {tool.name}({tool.server})", f"  {tool.description}"]

    properties = tool.parameters.get("properties")
    required = set(tool.required_parameters)
    if isinstance(properties, dict) and properties:
        rendered = []
        for param_name, param_schema in properties.items():
            schema = param_schema if isinstance(param_schema, dict) else {}
            entry = f"{param_name}: {_type_of(schema)}"
            if param_name in required:
                entry += " (required)"
            rendered.append(entry)
        lines.append("  params: " + ", ".join(rendered))
    else:
        lines.append("  params: none")

    if include_examples and tool.examples:
        lines.append("  examples: " + " | ".join(tool.examples[:3]))
    return "\n".join(lines)


def render_tool_json(tool: Tool) -> str:
    """Render one tool as full JSON Schema -- the unrouted-agent representation.

    Used by the ``all_tools`` benchmark baseline so its token count reflects what
    a real framework actually sends, not an artificially compressed version.
    """
    return json.dumps(
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "server": tool.server,
        },
        indent=2,
        sort_keys=False,
    )


def build_tool_prompt(
    tools: list[Tool],
    *,
    style: str = "compact",
    include_examples: bool = False,
    header: bool = True,
) -> str:
    """Build the tool-schema text block to inject into an agent system prompt.

    Parameters
    ----------
    tools:
        The routed subset -- **not** the full registry.
    style:
        ``"compact"`` (default) for the flattened rendering, or ``"json"`` for
        full JSON Schema per tool.
    include_examples:
        Append example utterances when the manifest supplied them.
    header:
        Prefix an instruction line. Disable when embedding into a larger prompt
        that already provides its own framing.
    """
    if style not in {"compact", "json"}:
        raise ValueError(f"style must be 'compact' or 'json', got {style!r}.")
    if not tools:
        return build_no_match_prompt()

    if style == "compact":
        body = "\n".join(
            render_tool_compact(t, include_examples=include_examples) for t in tools
        )
    else:
        body = "[\n" + ",\n".join(render_tool_json(t) for t in tools) + "\n]"

    if not header:
        return body

    plural = "tool" if len(tools) == 1 else "tools"
    intro = (
        f"You have access to the following {len(tools)} {plural}, pre-selected as "
        "relevant to the user's request. Choose the single best match and call it "
        "with the required parameters. If none of them fit, say so instead of "
        "guessing."
    )
    return f"{intro}\n\n{body}"


def build_no_match_prompt() -> str:
    """Prompt block used when the confidence gate reports no confident match."""
    return (
        "No tool in the connected MCP servers matched this request with sufficient "
        "confidence. Tell the user you cannot handle this request with the "
        "available tools rather than calling an unrelated tool."
    )


def estimate_tokens(text: str) -> int:
    """Approximate token count for ``text``.

    Uses ``tiktoken`` when installed for a real BPE count; otherwise applies a
    word/character heuristic. The benchmark records which method was used, since
    the reduction percentage depends on it.
    """
    if not text:
        return 0
    try:  # pragma: no cover - depends on optional dependency
        import tiktoken  # type: ignore

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # noqa: BLE001
        # ~4 characters per token is the standard English approximation; we take
        # the max against whitespace-token count so dense JSON isn't undercounted.
        words = len(text.split())
        return max(int(len(text) / 4), int(words * 1.3), 1)


def tokens_for_tools(tools: Iterable[Tool], *, style: str = "compact") -> int:
    """Token count of the prompt block for ``tools`` (no header)."""
    tool_list = list(tools)
    if not tool_list:
        return 0
    return estimate_tokens(build_tool_prompt(tool_list, style=style, header=False))
