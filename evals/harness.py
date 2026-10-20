"""
Runs the real gget agent on one request, outside iChatBio, and records what it did.

- Uses the iChatBio SDK's InMemoryResponseChannel, so every reply, process log and artifact the
  tools send is captured exactly as iChatBio would receive it.
- Records every tool call (name, arguments, output) by wrapping the LangChain graph that
  agent.py builds. agent.py itself is not modified.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import agent as agent_module  # noqa: E402  (src/agent.py)
from ichatbio.agent_response import (  # noqa: E402
    ArtifactResponse,
    DirectResponse,
    ProcessLogResponse,
    ResponseContext,
)
from ichatbio.test_utils import InMemoryResponseChannel  # noqa: E402

TERMINAL_TOOLS = {"finish", "abort"}
TOOL_OUTPUT_MAX_CHARS = 6000


@dataclass
class RecordedCall:
    name: str
    args: dict
    output: str | None = None
    id: str | None = None


@dataclass
class AgentRun:
    request: str
    calls: list[RecordedCall] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    seconds: float = 0.0

    @property
    def work_calls(self) -> list[RecordedCall]:
        """Tool calls excluding finish/abort."""
        return [c for c in self.calls if c.name not in TERMINAL_TOOLS]

    @property
    def outcome(self) -> str:
        """'finish', 'abort', 'error', or 'none' (loop ended without a terminal tool)."""
        if self.error:
            return "error"
        for c in reversed(self.calls):
            if c.name in TERMINAL_TOOLS:
                return c.name
        return "none"

    @property
    def final_answer(self) -> str:
        return self.replies[-1] if self.replies else ""

    def tool_outputs_text(self) -> str:
        return "\n".join(c.output or "" for c in self.work_calls)


# ── Recording wrapper around the LangChain agent graph ────────────────────────

_current_run: AgentRun | None = None
_original_create_agent = agent_module.create_agent


def _record_messages(messages: list[Any]) -> None:
    run = _current_run
    if run is None:
        return
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            run.calls.append(RecordedCall(name=call["name"], args=dict(call.get("args") or {}), id=call.get("id")))
        if getattr(msg, "type", None) == "tool":
            text = str(msg.content)[:TOOL_OUTPUT_MAX_CHARS]
            match = next((c for c in run.calls if c.id and c.id == getattr(msg, "tool_call_id", None)), None)
            if match is None:  # fall back to the latest call of that name with no output yet
                match = next((c for c in reversed(run.calls) if c.name == msg.name and c.output is None), None)
            if match is not None:
                match.output = text


class _RecordingGraph:
    def __init__(self, graph):
        self._graph = graph

    async def astream(self, *args, **kwargs):
        async for update in self._graph.astream(*args, **kwargs):
            if isinstance(update, dict):
                for state in update.values():
                    if isinstance(state, dict):
                        _record_messages(state.get("messages", []))
            yield update

    async def ainvoke(self, *args, **kwargs):
        result = await self._graph.ainvoke(*args, **kwargs)
        _record_messages(result.get("messages", []) if isinstance(result, dict) else [])
        return result

    def __getattr__(self, name):
        return getattr(self._graph, name)


def _recording_create_agent(*args, **kwargs):
    return _RecordingGraph(_original_create_agent(*args, **kwargs))


agent_module.create_agent = _recording_create_agent


# ── Running one request ───────────────────────────────────────────────────────

def _build_params(query: str, sequence: str | None):
    """Use the entrypoint's `sequence` field if agent.py has one; otherwise append it to the request."""
    params_cls = agent_module.EntrypointParameters
    if sequence and "sequence" in params_cls.model_fields:
        return params_cls(user_request=query, sequence=sequence), query
    request = f"{query}\n\n{sequence}" if sequence else query
    return params_cls(user_request=request), request


async def run_agent(query: str, sequence: str | None = None, timeout: float = 900) -> AgentRun:
    """Runs the agent once and returns everything it did."""
    global _current_run
    params, request = _build_params(query, sequence)
    run = AgentRun(request=request)
    _current_run = run

    messages: list = []
    context = ResponseContext(InMemoryResponseChannel(messages))
    metadata = json.loads(os.getenv("EVAL_AGENT_METADATA", "null"))  # if your util needs iChatBio metadata

    start = time.monotonic()
    try:
        await asyncio.wait_for(
            agent_module.GgetNucleotideAgent().run(context, request, "process_request", params, metadata),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        run.error = f"timed out after {timeout:.0f}s"
    except Exception as e:  # recursion limit, LLM errors, crashes
        run.error = f"{type(e).__name__}: {e}"
        run.traceback = traceback.format_exc()
    finally:
        run.seconds = round(time.monotonic() - start, 1)
        _current_run = None

    for m in messages:
        if isinstance(m, DirectResponse) and m.text:
            run.replies.append(m.text)
        elif isinstance(m, ProcessLogResponse) and m.text:
            run.logs.append(m.text)
        elif isinstance(m, ArtifactResponse):
            run.artifacts.append(m.description or m.mimetype or "artifact")
    return run
