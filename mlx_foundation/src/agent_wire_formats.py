"""
Wire formats for agentic training / inference.

- ``dingo``: ``<|channel>thought`` + ``write_file: ... | python: ...`` (legacy Dingo track)
- ``omlx_claude``: ``<|channel>thought`` + Gemma ``<|tool_call>call:Read{...}<tool_call|>`` (oMLX / Claude Code)
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

GEMMA_TOOL_OPEN = "<|tool_call>"
GEMMA_TOOL_CLOSE = "<tool_call|>"

# Claude Code tool names (what oMLX maps to Anthropic tool_use).
CLAUDE_TOOL_NAMES = frozenset({"Read", "Write", "Edit", "Bash", "Glob", "Grep", "ListDir"})

DINGO_TO_CLAUDE = {
    "read_file": "Read",
    "write_file": "Write",
    "list_dir": "Glob",
    "python": "Bash",
    "none": "none",
}

CLAUDE_TO_SANDBOX = {
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "edit",
    "Glob": "list_dir",
    "Grep": "read_file",
    "ListDir": "list_dir",
    "Bash": "bash",
    "none": "none",
}


def _gemma_quote(value: str) -> str:
    return f'<|"|>{value}<|"|>'


def _format_gemma_args(args: Dict[str, Any]) -> str:
    """Render ``{key: value, ...}`` in Gemma tool-call arg style."""
    if not args:
        return "{}"
    parts: List[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            parts.append(f"{key}: {_gemma_quote(value)}")
        elif isinstance(value, bool):
            parts.append(f"{key}: {str(value).lower()}")
        elif value is None:
            parts.append(f"{key}: null")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value}")
        else:
            parts.append(f"{key}: {_gemma_quote(json.dumps(value, ensure_ascii=False))}")
    return "{" + ", ".join(parts) + "}"


def format_omlx_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """One Gemma tool invocation (markers included)."""
    return f"{GEMMA_TOOL_OPEN}call:{name}{_format_gemma_args(arguments)}{GEMMA_TOOL_CLOSE}"


def _parse_action_input(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return text
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text
    return text


def dingo_action_to_claude_args(action_type: str, action_input: Any) -> Tuple[str, Dict[str, Any]]:
    """Map a Dingo sandbox action to Claude tool name + arguments dict."""
    if action_type == "none":
        return "none", {}

    if action_type == "read_file":
        path = action_input if isinstance(action_input, str) else str(
            (action_input or {}).get("path") or (action_input or {}).get("file_path") or ""
        )
        return "Read", {"file_path": path.strip()}

    if action_type == "write_file":
        if isinstance(action_input, dict):
            path = str(action_input.get("path") or action_input.get("file_path") or "")
            content = str(action_input.get("content", ""))
        else:
            path, content = "", str(action_input)
            if isinstance(action_input, str) and ":" in action_input and not action_input.strip().startswith("{"):
                path, content = action_input.split(":", 1)
                path = path.strip()
        return "Write", {"file_path": path, "content": content}

    if action_type == "list_dir":
        return "Glob", {"pattern": "**/*", "path": "."}

    if action_type == "python":
        code = action_input if isinstance(action_input, str) else str(action_input)
        command = (
            "PYTHONPATH=. python3 <<'PY'\n"
            f"{code.rstrip()}\n"
            "PY"
        )
        return "Bash", {"command": command}

    return action_type, {"input": action_input}


def dingo_turn_to_omlx_call(action_type: str, action_input: Any) -> str:
    name, args = dingo_action_to_claude_args(action_type, action_input)
    if name == "none":
        return "none:"
    return format_omlx_tool_call(name, args)


def _try_edit_from_context(
    path: str,
    new_content: str,
    prior_content: Optional[str],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """If we have file content before write, emit Edit with a single old/new pair."""
    if prior_content is None or prior_content == new_content:
        return None
    if prior_content not in new_content and new_content not in prior_content:
        return None
    if len(new_content) - len(prior_content) > 8000:
        return None
    import difflib

    sm = difflib.SequenceMatcher(None, prior_content, new_content)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) < 2000 and (j2 - j1) < 2000:
            old = prior_content[i1:i2]
            new = new_content[j1:j2]
            if old and new and old != new:
                return (
                    "Edit",
                    {
                        "file_path": path,
                        "old_string": old,
                        "new_string": new,
                    },
                )
    return None


def turns_to_omlx_actions(turns: List[Dict[str, Any]]) -> str:
    """Build pipe-separated oMLX tool-call string from trajectory turns."""
    parts: List[str] = []
    file_cache: Dict[str, str] = {}

    for turn in turns:
        action = turn.get("action") or {}
        atype = action.get("type", "none")
        ainput = _parse_action_input(action.get("input", ""))

        if atype == "read_file":
            path = ainput if isinstance(ainput, str) else str(
                (ainput or {}).get("path") or (ainput or {}).get("file_path") or ""
            )
            obs = turn.get("observation") or {}
            content = obs.get("content") or obs.get("stdout") or ""
            if content:
                file_cache[path.strip()] = content
            parts.append(dingo_turn_to_omlx_call(atype, ainput))
            continue

        if atype == "write_file":
            if isinstance(ainput, dict):
                path = str(ainput.get("path") or ainput.get("file_path") or "")
                content = str(ainput.get("content", ""))
            else:
                path, content = "", str(ainput)
            edit = _try_edit_from_context(path, content, file_cache.get(path))
            if edit:
                parts.append(format_omlx_tool_call(edit[0], edit[1]))
                file_cache[path] = content
            else:
                parts.append(dingo_turn_to_omlx_call(atype, ainput))
                file_cache[path] = content
            continue

        if atype != "none":
            parts.append(dingo_turn_to_omlx_call(atype, ainput))

    return " | ".join(parts)


def sample_to_omlx_training_fields(sample: Dict[str, Any]) -> Dict[str, str]:
    """Return ``thought`` and ``actions`` strings in oMLX/Gemma tool-call form."""
    turns = sample.get("turns") or []
    if turns:
        thoughts = [
            (t.get("thought") or "").strip()
            for t in turns
            if (t.get("action") or {}).get("type") != "none"
        ]
        thought = thoughts[-1] if thoughts else (sample.get("thought") or "").split(" | ")[-1]
        actions = turns_to_omlx_actions(turns)
    else:
        thought = sample.get("thought", "")
        actions_parts: List[str] = []
        for chunk in (sample.get("actions") or "").split(" | "):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("call:") or chunk.startswith(GEMMA_TOOL_OPEN):
                actions_parts.append(chunk)
                continue
            if ":" in chunk:
                atype, ainput = chunk.split(":", 1)
                actions_parts.append(
                    dingo_turn_to_omlx_call(atype.strip(), ainput.strip())
                )
            else:
                actions_parts.append(chunk)
        actions = " | ".join(actions_parts)
    return {"thought": thought.strip(), "actions": actions}


def convert_sample_for_omlx_training(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a curated row with oMLX ``actions`` (+ metadata)."""
    out = dict(sample)
    fields = sample_to_omlx_training_fields(sample)
    out["actions"] = fields["actions"]
    out["thought"] = fields["thought"]
    out["wire_format"] = "omlx_claude"
    return out


def _parse_gemma_call_args(args_str: str) -> Dict[str, Any]:
    """Best-effort parse of ``{key: <|"|>val<|"|>, ...}`` tool args."""
    inner = args_str.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]

    def _unquote_gemma(s: str) -> str:
        s = s.strip()
        if s.startswith('<|"|>') and s.endswith('<|"|>'):
            return s[5:-5]
        return s.strip("'\"")

    result: Dict[str, Any] = {}
    if not inner.strip():
        return result

    try:
        parsed = json.loads("{" + inner + "}")
        if isinstance(parsed, dict):
            return {k: _unquote_gemma(v) if isinstance(v, str) else v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    for piece in re.split(r",\s*(?=\w+\s*:)", inner):
        if ":" not in piece:
            continue
        key, val = piece.split(":", 1)
        result[key.strip()] = _unquote_gemma(val)
    return result


def sandbox_action_type(claude_or_dingo_type: str) -> str:
    """Resolve tool name to internal sandbox executor type."""
    key = claude_or_dingo_type.strip()
    if key in CLAUDE_TO_SANDBOX:
        return CLAUDE_TO_SANDBOX[key]
    if key in DINGO_TO_CLAUDE:
        return key
    return key


def parse_omlx_agent_step(raw: str) -> Optional[Dict[str, Any]]:
    """
    Parse one agent step: channel thought + ``call:Tool{args}``.

    Returns dict with thought, action_type (Claude name), action_input (dict/str).
    """
    text = raw or ""
    thought = ""
    if "<|channel>thought" in text:
        after = text.split("<|channel>thought", 1)[-1]
        if "<channel|>" in after:
            thought, rest = after.split("<channel|>", 1)
            thought = thought.strip()
            text = rest
        else:
            text = after

    tool_text = text
    if GEMMA_TOOL_OPEN in tool_text:
        idx = tool_text.find(GEMMA_TOOL_OPEN)
        tool_text = tool_text[idx + len(GEMMA_TOOL_OPEN) :]
        if GEMMA_TOOL_CLOSE in tool_text:
            tool_text = tool_text.split(GEMMA_TOOL_CLOSE, 1)[0]

    match = re.search(r"call:(\w+)(\{[\s\S]*\})", tool_text)
    if not match:
        if "none" in tool_text.lower() and "call:" not in tool_text:
            return {
                "thought": thought or "Done.",
                "action_type": "none",
                "action_input": "",
                "final_answer": tool_text.strip() or "Task complete.",
            }
        return None

    name = match.group(1)
    args_str = match.group(2)
    arguments = _parse_gemma_call_args(args_str)

    if not isinstance(arguments, dict):
        arguments = {"raw": arguments}

    return {
        "thought": thought,
        "action_type": name,
        "action_input": arguments,
        "final_answer": "",
    }


OMLX_AGENT_SYSTEM_APPENDIX = """
OUTPUT FORMAT (oMLX / Claude Code — mandatory):
- Put reasoning ONLY inside <|channel>thought ... <channel|>.
- After <channel|>, emit EXACTLY ONE tool per turn using Gemma markers:
  <|tool_call>call:ToolName{arg: <|"|>value<|"|>, ...}<tool_call|>
- Tool names: Read, Write, Edit, Bash, Glob (not write_file/python).
- Read: call:Read{file_path: <|"|>src/foo.py<|"|>}
- Write: call:Write{file_path: <|"|>src/foo.py<|"|>, content: <|"|>...full file...<|"|>}
- Edit: call:Edit{file_path: <|"|>src/foo.py<|"|>, old_string: <|"|>...<|"|>, new_string: <|"|>...<|"|>}
- Bash: call:Bash{command: <|"|>PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'<|"|>}
- When finished: <|tool_call>call:Done{}<tool_call|> or action none with final_answer in thought.
- No markdown fences. No JSON action_type blobs. No write_file: pipe syntax.
"""
