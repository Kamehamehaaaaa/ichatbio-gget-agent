"""
Shared helper for running blocking gget calls from async tools.

Runs the call in a worker thread, sends periodic progress messages to iChatBio while it runs,
and gives up after a timeout. Use it from any tool that calls gget.
"""

import asyncio
import time
from typing import Any, Callable

from ichatbio.agent_response import IChatBioAgentProcess


class GgetTimeout(Exception):
    pass


async def run_gget(
    process: IChatBioAgentProcess,
    fn: Callable[..., Any],
    *args,
    label: str = "gget",
    heartbeat_seconds: int = 15,
    timeout_seconds: int = 300,
    **kwargs,
) -> Any:
    """
    Runs fn(*args, **kwargs) in a thread. Logs "still running" every `heartbeat_seconds`.
    Raises GgetTimeout after `timeout_seconds`; re-raises any exception from fn.
    """
    start = time.monotonic()
    task = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    elapsed = 0
    while True:
        done, _ = await asyncio.wait({task}, timeout=heartbeat_seconds)
        if done:
            break
        elapsed += heartbeat_seconds
        if elapsed >= timeout_seconds:
            task.cancel()  # stops waiting; the thread finishes on its own
            raise GgetTimeout(f"{label} did not finish within {timeout_seconds}s")
        minutes, seconds = divmod(elapsed, 60)
        await process.log(f"{label} still running ({minutes}m {seconds:02d}s elapsed)...")

    result = task.result()  # re-raises exceptions from fn
    took = time.monotonic() - start
    if took >= heartbeat_seconds:
        await process.log(f"{label} finished in {took:.0f}s")
    return result


def normalize_species(species: str) -> str:
    """'Homo sapiens' / 'homo sapiens' -> 'homo_sapiens'. 'human' and 'mouse' pass through."""
    return species.strip().lower().replace(" ", "_")