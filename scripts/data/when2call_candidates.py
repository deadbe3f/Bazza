"""Auditable four-action When2Call candidate conversion, unchanged from v0.4."""

import copy
import json
import re

from common import _membership_digest as digest

VERSION = "when2call-fourway-v0.1"


def classify(text):
    if text.startswith("<TOOLCALL>"):
        return "tool_call"
    if re.search(
        r"\b(unable|can't|cannot|can not|don't have|do not have|beyond my capabilities)\b",
        text,
        re.I,
    ):
        return "cannot_answer"
    if re.search(
        r"\b(could you|can you|please provide|please specify|i need|i.?ll need|which specific|in which|for what specific|just to confirm|just to clarify|to clarify)\b",
        text,
        re.I,
    ):
        return "request_for_info"
    if re.match(r"(what|which|how many)\b", text, re.I) and text.rstrip().endswith("?"):
        return "request_for_info"
    return "direct"


def plain_call(text):
    if not text.startswith("<TOOLCALL>"):
        return text
    if not text.endswith("</TOOLCALL>"):
        raise ValueError("Unclosed native tool-call wrapper")
    calls = json.loads(text[len("<TOOLCALL>") : -len("</TOOLCALL>")])
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Four-action view requires exactly one native tool call")
    return json.dumps(calls[0], ensure_ascii=False)


def words(value):
    return set(re.findall(r"[a-z0-9]+", str(value).lower())) - {
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "from",
        "with",
        "and",
        "in",
        "me",
        "my",
        "please",
    }


def relevant_tool(query, tools):
    # Deterministic retrieval for a negative candidate, without consulting gold.
    query_words = words(query)
    return max(
        tools,
        key=lambda t: (
            len(query_words & words(t["name"] + " " + t.get("description", ""))),
            digest(t),
        ),
    )


def parameter_value(schema):
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if schema.get("enum"):
        return copy.deepcopy(schema["enum"][0])
    kind = schema.get("type", "string")
    if kind in ("object", "dict"):
        props = schema.get("properties", {})
        return {
            k: parameter_value(props.get(k, {}))
            for k in schema.get("required", list(props))
        }
    if kind in ("array", "list"):
        return [parameter_value(schema.get("items", {}))]
    if kind in ("integer", "int"):
        return 1
    if kind in ("number", "float"):
        return 1.0
    if kind in ("boolean", "bool"):
        return True
    return "example"


def added_candidate(policy, row):
    query, tools = row["state"]["question"], row["state"]["tools"]
    tool = (
        relevant_tool(query, tools)
        if tools
        else {
            "name": "execute_request",
            "parameters": {
                "properties": {"request": {"type": "string", "default": query}},
                "required": ["request"],
            },
        }
    )
    schema = tool.get("parameters", {})
    props = schema.get("properties", {})
    required = tool.get("required", schema.get("required", list(props)))
    selector = int(digest([VERSION, row["id"], policy])[:8], 16)
    if policy == "tool_call":
        # Defaults are deliberately not treated as supplied user information.
        # This candidate is only added when the native preferred response is
        # non-call; it may guess a missing argument or use an irrelevant tool.
        args = {k: parameter_value(props.get(k, {})) for k in required}
        text = json.dumps({"name": tool["name"], "arguments": args}, ensure_ascii=False)
        method = (
            "closest available tool; schema defaults or typed example values"
            if tools
            else "invented unavailable execute_request tool; no tools exist in this menu"
        )
    elif policy == "request_for_info":
        name = (
            (required or list(props) or ["request details"])[
                selector % max(1, len(required or list(props)))
            ]
            if tools
            else "preferred information source"
        )
        detail = str(name).replace("_", " ")
        templates = (
            "Before I proceed, could you confirm the {detail} you want me to use?",
            "Could you please specify the {detail} for this request?",
            "To assist you with this request, please provide the {detail}.",
            "Which {detail} should I use to carry out your request?",
        )
        text = templates[selector % len(templates)].format(detail=detail)
        method = "clarification about a closest-tool argument; possibly unnecessary or irrelevant"
    elif policy == "cannot_answer":
        templates = (
            "I'm sorry, but I can't complete that request with the tools currently available.",
            "The available tools don't support that request, so I'm unable to carry it out.",
            "I don't have a tool that can perform the requested operation. I can't complete it here.",
            "Apologies, I'm unable to carry out this request using the capabilities available to me.",
        )
        text = templates[selector % len(templates)]
        method = "unavailable-capability response template"
    else:
        # Explicit fabricated completion, never a new gold label. Prefer the
        # original (usually much more specific) direct negative when present.
        quoted = query.rstrip(".?!")
        templates = (
            'Done. I have completed your request: "{query}".',
            'Your request has been completed successfully: "{query}".',
            'I have carried out the requested operation: "{query}".',
            'All set. The requested result is ready: "{query}".',
        )
        text = templates[selector % len(templates)].format(query=quoted)
        method = "unsupported completion claim referencing the request"
    return text, {
        "origin": "synthetic_template",
        "author": "Codex",
        "method": method,
        "generator_version": VERSION,
        "template_selector": selector,
    }
