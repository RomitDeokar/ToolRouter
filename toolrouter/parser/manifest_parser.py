"""MCP manifest parsing.

This module owns the canonical :class:`Tool` type used by every other module in
the package, plus :func:`parse_manifest`, which turns an arbitrary MCP manifest
into ``list[Tool]``.

Design rule (see ``ARCHITECTURE.md``): the parser must be *generic*. Real MCP
servers disagree about field names -- ``parameters`` vs ``inputSchema`` vs
``input_schema``, ``tools`` vs ``functions``, a server name at the root vs one
per tool vs none at all. Rather than hardcoding one vendor's shape, we resolve
each field through an ordered list of aliases and fail loudly when a *required*
field genuinely cannot be found.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Tool",
    "ManifestError",
    "parse_manifest",
    "parse_manifest_dict",
    "load_manifest_document",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ManifestError(ValueError):
    """Raised when a manifest cannot be parsed into well-formed ``Tool`` objects.

    Deliberately a specific exception type carrying a human-readable message,
    so callers never have to interpret a bare ``KeyError``/``TypeError`` coming
    out of dictionary access.
    """


# --------------------------------------------------------------------------- #
# Core type
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    """A single callable tool exposed by an MCP server.

    Attributes
    ----------
    name:
        Tool identifier as the agent would call it, e.g. ``"search_restaurants"``.
    description:
        Natural-language description from the manifest.
    parameters:
        Raw JSON Schema object for the tool's arguments. Kept verbatim so the
        prompt builder can emit a faithful schema.
    server:
        Logical MCP server the tool belongs to, e.g. ``"food"``. Used for
        per-server filtering and (future) hierarchical routing.
    examples:
        Optional sample invocations / example utterances from the manifest.
    """

    name: str
    description: str
    parameters: dict
    server: str
    examples: list[str] = field(default_factory=list)

    # -- embedding text ---------------------------------------------------- #
    def to_embedding_text(self) -> str:
        """Return the text used to embed this tool.

        Per the design spec this includes ``name`` + ``description`` +
        *parameter names* -- not the description alone. Parameter names carry
        disambiguating signal that descriptions often miss: ``search_restaurants``
        (``cuisine``, ``location``) and ``search_products`` (``category``,
        ``brand``) can have near-identical descriptions but very different
        argument vocabularies.

        Examples
        --------
        >>> t = Tool(
        ...     name="search_restaurants",
        ...     description="Search restaurants by cuisine.",
        ...     parameters={"properties": {"cuisine": {"type": "string"}}},
        ...     server="food",
        ... )
        >>> t.to_embedding_text()
        'search_restaurants - Search restaurants by cuisine. - cuisine'
        """
        parts = [
            self.name,
            _humanize_identifier(self.name),
            self.description,
            ", ".join(self.parameter_names),
        ]
        if self.examples:
            parts.append(" | ".join(self.examples))
        # Drop the humanized alias when it adds nothing over the raw name.
        if parts[1] == parts[0]:
            parts.pop(1)
        return " - ".join(p for p in parts if p)

    # -- convenience ------------------------------------------------------- #
    @property
    def parameter_names(self) -> list[str]:
        """Top-level parameter names declared in the tool's JSON Schema."""
        props = self.parameters.get("properties")
        if isinstance(props, Mapping):
            return [str(k) for k in props]
        return []

    @property
    def required_parameters(self) -> list[str]:
        """Parameter names listed as required in the tool's JSON Schema."""
        required = self.parameters.get("required")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            return [str(r) for r in required]
        return []

    @property
    def qualified_name(self) -> str:
        """``server.name`` -- unique even if two servers share a tool name."""
        return f"{self.server}.{self.name}" if self.server else self.name

    def to_dict(self) -> dict:
        """JSON-serialisable view of the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "server": self.server,
            "examples": list(self.examples),
        }


# --------------------------------------------------------------------------- #
# Field aliases -- the whole point of "generic"
# --------------------------------------------------------------------------- #
#: Keys that may hold the list of tools in a manifest document.
_TOOL_LIST_KEYS: tuple[str, ...] = ("tools", "functions", "actions", "capabilities")
#: Keys that may hold the tool name.
_NAME_KEYS: tuple[str, ...] = ("name", "tool_name", "toolName", "id", "function_name")
#: Keys that may hold the tool description.
_DESCRIPTION_KEYS: tuple[str, ...] = (
    "description",
    "desc",
    "summary",
    "doc",
    "documentation",
)
#: Keys that may hold the JSON Schema for the tool's arguments.
_PARAM_KEYS: tuple[str, ...] = (
    "parameters",
    "inputSchema",
    "input_schema",
    "schema",
    "args_schema",
    "arguments",
    "input",
)
#: Keys that may hold the owning server name.
_SERVER_KEYS: tuple[str, ...] = (
    "server",
    "server_name",
    "serverName",
    "namespace",
    "source",
    "provider",
    "mcp_server",
)
#: Keys that may hold example utterances / sample calls.
_EXAMPLE_KEYS: tuple[str, ...] = ("examples", "sample_queries", "samples", "usage")


def _first_present(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-empty value among ``keys``, else ``None``."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is not None and value != "" and value != {} and value != []:
                return value
    return None


def _humanize_identifier(identifier: str) -> str:
    """``search_restaurants`` / ``searchRestaurants`` -> ``search restaurants``."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier)
    spaced = re.sub(r"[_\-.:/]+", " ", spaced)
    return re.sub(r"\s+", " ", spaced).strip().lower()


def _coerce_examples(value: Any) -> list[str]:
    """Normalise whatever the manifest put under ``examples`` into ``list[str]``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [json.dumps(value, separators=(",", ":"), sort_keys=True)]
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping):
                # Prefer a human-readable field if the example is an object.
                text = _first_present(item, ("query", "prompt", "text", "utterance"))
                out.append(
                    str(text)
                    if text is not None
                    else json.dumps(item, separators=(",", ":"), sort_keys=True)
                )
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _coerce_parameters(value: Any, *, tool_label: str) -> dict:
    """Normalise the parameter schema into a JSON-Schema-shaped ``dict``.

    A missing schema is legal (zero-argument tools exist) and becomes
    ``{"type": "object", "properties": {}}``. A schema of the wrong *type*
    is not legal and raises, because silently discarding it would produce a
    ``Tool`` whose embedding text is missing its parameter signal.
    """
    if value is None:
        return {"type": "object", "properties": {}}
    if isinstance(value, Mapping):
        schema = dict(value)
        # Some servers emit {"cuisine": {"type": "string"}} without the
        # surrounding {"type": "object", "properties": {...}} envelope.
        if (
            schema
            and "properties" not in schema
            and "type" not in schema
            and all(isinstance(v, Mapping) for v in schema.values())
        ):
            return {"type": "object", "properties": schema}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        if not isinstance(schema["properties"], Mapping):
            raise ManifestError(
                f"{tool_label}: 'properties' must be an object, got "
                f"{type(schema['properties']).__name__}."
            )
        return schema
    raise ManifestError(
        f"{tool_label}: parameter schema must be an object, got "
        f"{type(value).__name__}."
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_manifest_document(path_or_url: str, *, timeout: float = 10.0) -> Any:
    """Load raw JSON from a local path or an ``http(s)`` URL.

    Raises
    ------
    ManifestError
        If the location cannot be read or does not contain valid JSON.
    """
    if not isinstance(path_or_url, str) or not path_or_url.strip():
        raise ManifestError("Manifest location must be a non-empty string.")

    location = path_or_url.strip()
    if location.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(location, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ManifestError(f"Could not fetch manifest from {location!r}: {exc}") from exc
    else:
        if not os.path.isfile(location):
            raise ManifestError(f"Manifest file not found: {location!r}")
        try:
            with open(location, encoding="utf-8") as handle:
                raw = handle.read()
        except OSError as exc:
            raise ManifestError(f"Could not read manifest file {location!r}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest at {location!r} is not valid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _iter_tool_entries(
    document: Any, *, inherited_server: str | None = None
) -> list[tuple[Mapping[str, Any], str | None]]:
    """Flatten a manifest document into ``[(tool_entry, server_hint), ...]``.

    Handles the four shapes seen in the wild:

    1. ``{"server": "food", "tools": [...]}``           -- single server
    2. ``[{...}, {...}]``                                 -- bare tool list
    3. ``[{"server": "food", "tools": [...]}, ...]``    -- list of servers
    4. ``{"servers": {"food": {"tools": [...]}}}``      -- server map
    """
    entries: list[tuple[Mapping[str, Any], str | None]] = []

    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        for item in document:
            if isinstance(item, Mapping) and _first_present(item, _TOOL_LIST_KEYS):
                entries.extend(_iter_tool_entries(item, inherited_server=inherited_server))
            elif isinstance(item, Mapping):
                entries.append((item, inherited_server))
            else:
                raise ManifestError(
                    f"Expected tool entries to be objects, found {type(item).__name__}."
                )
        return entries

    if not isinstance(document, Mapping):
        raise ManifestError(
            "Manifest must be a JSON object or array, got "
            f"{type(document).__name__}."
        )

    server_hint = _first_present(document, _SERVER_KEYS) or inherited_server
    server_hint = str(server_hint) if server_hint is not None else None

    # Shape 4: a mapping of server-name -> server document.
    servers = document.get("servers")
    if isinstance(servers, Mapping):
        for server_name, server_doc in servers.items():
            entries.extend(_iter_tool_entries(server_doc, inherited_server=str(server_name)))
        return entries
    if isinstance(servers, Sequence) and not isinstance(servers, (str, bytes)):
        for server_doc in servers:
            entries.extend(_iter_tool_entries(server_doc, inherited_server=server_hint))
        return entries

    tool_list = _first_present(document, _TOOL_LIST_KEYS)
    if tool_list is None:
        # Distinguish "the key is missing entirely" from "the key is present but
        # empty" -- _first_present() treats both as absent, and reporting a
        # missing key while that key is visibly in the document is confusing.
        present_but_empty = [key for key in _TOOL_LIST_KEYS if key in document]
        if present_but_empty:
            raise ManifestError(
                f"Manifest declares {present_but_empty[0]!r} but it is empty: the "
                "manifest parsed successfully and contains zero tools."
            )
        raise ManifestError(
            "Manifest contains no tool list. Expected one of "
            f"{list(_TOOL_LIST_KEYS)} at the top level, found keys "
            f"{sorted(document.keys())}."
        )

    # A mapping of tool-name -> tool body is also legal.
    if isinstance(tool_list, Mapping):
        for tool_name, body in tool_list.items():
            if isinstance(body, Mapping):
                merged = dict(body)
                merged.setdefault("name", tool_name)
                entries.append((merged, server_hint))
            else:
                raise ManifestError(
                    f"Tool {tool_name!r} must map to an object, got {type(body).__name__}."
                )
        return entries

    if not isinstance(tool_list, Sequence) or isinstance(tool_list, (str, bytes)):
        raise ManifestError(
            f"Tool list must be an array or object, got {type(tool_list).__name__}."
        )

    for item in tool_list:
        if not isinstance(item, Mapping):
            raise ManifestError(
                f"Expected tool entries to be objects, found {type(item).__name__}."
            )
        # Nested server documents inside a tool list.
        if _first_present(item, _TOOL_LIST_KEYS):
            entries.extend(_iter_tool_entries(item, inherited_server=server_hint))
        else:
            entries.append((item, server_hint))
    return entries


def _build_tool(
    entry: Mapping[str, Any],
    *,
    server_hint: str | None,
    default_server: str,
    position: int,
) -> Tool:
    label = f"Tool #{position}"

    raw_name = _first_present(entry, _NAME_KEYS)
    if raw_name is None:
        raise ManifestError(
            f"{label} is missing a required 'name' field (accepted aliases: "
            f"{list(_NAME_KEYS)}). Offending entry keys: {sorted(entry.keys())}."
        )
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ManifestError(f"{label}: 'name' must be a non-empty string, got {raw_name!r}.")
    name = raw_name.strip()
    label = f"Tool {name!r}"

    raw_description = _first_present(entry, _DESCRIPTION_KEYS)
    if raw_description is None:
        raise ManifestError(
            f"{label} is missing a required 'description' field (accepted aliases: "
            f"{list(_DESCRIPTION_KEYS)}). A tool with no description cannot be "
            "embedded meaningfully."
        )
    if not isinstance(raw_description, str) or not raw_description.strip():
        raise ManifestError(
            f"{label}: 'description' must be a non-empty string, got {raw_description!r}."
        )

    parameters = _coerce_parameters(_first_present(entry, _PARAM_KEYS), tool_label=label)

    raw_server = _first_present(entry, _SERVER_KEYS) or server_hint or default_server
    server = str(raw_server).strip()
    if not server:
        raise ManifestError(f"{label}: resolved server name is empty.")

    return Tool(
        name=name,
        description=raw_description.strip(),
        parameters=parameters,
        server=server,
        examples=_coerce_examples(_first_present(entry, _EXAMPLE_KEYS)),
    )


def parse_manifest_dict(
    document: Any,
    *,
    default_server: str = "default",
    allow_duplicate_names: bool = False,
) -> list[Tool]:
    """Parse an already-loaded manifest document into ``list[Tool]``.

    Parameters
    ----------
    document:
        Parsed JSON (``dict`` or ``list``).
    default_server:
        Server name assigned to tools that declare none anywhere up the tree.
    allow_duplicate_names:
        When ``False`` (default), a repeated tool name raises. Duplicate names
        would collide in the vector index, which keys candidates by name.
    """
    entries = _iter_tool_entries(document)
    if not entries:
        raise ManifestError("Manifest parsed successfully but contains zero tools.")

    tools: list[Tool] = []
    seen: dict[str, int] = {}
    for position, (entry, server_hint) in enumerate(entries, start=1):
        tool = _build_tool(
            entry,
            server_hint=server_hint,
            default_server=default_server,
            position=position,
        )
        if tool.name in seen and not allow_duplicate_names:
            raise ManifestError(
                f"Duplicate tool name {tool.name!r} (entries #{seen[tool.name]} and "
                f"#{position}). Tool names must be unique because the index is "
                "keyed by name; pass allow_duplicate_names=True to override."
            )
        seen[tool.name] = position
        tools.append(tool)
    return tools


def parse_manifest(
    path_or_url: str,
    *,
    default_server: str = "default",
    allow_duplicate_names: bool = False,
    timeout: float = 10.0,
) -> list[Tool]:
    """Parse an MCP manifest from a local path or ``http(s)`` URL.

    The parser is manifest-agnostic: field names are resolved through alias
    lists, and several document layouts are supported.

    Examples
    --------
    Single-server document::

        {
          "server": "food",
          "tools": [
            {"name": "search_restaurants",
             "description": "Search restaurants by cuisine.",
             "parameters": {"type": "object",
                            "properties": {"cuisine": {"type": "string"}}}}
          ]
        }

    Multi-server document using MCP's ``inputSchema`` spelling::

        [
          {"server": "github",
           "tools": [{"name": "create_issue",
                      "description": "Open a GitHub issue.",
                      "inputSchema": {"properties": {"title": {"type": "string"}}}}]},
          {"server": "slack",
           "tools": [{"name": "post_message",
                      "description": "Post a message to a channel.",
                      "inputSchema": {"properties": {"channel": {"type": "string"}}}}]}
        ]

    Raises
    ------
    ManifestError
        If the manifest cannot be read, is not valid JSON, contains no tools,
        or has a tool entry missing a required field.
    """
    document = load_manifest_document(path_or_url, timeout=timeout)
    return parse_manifest_dict(
        document,
        default_server=default_server,
        allow_duplicate_names=allow_duplicate_names,
    )
