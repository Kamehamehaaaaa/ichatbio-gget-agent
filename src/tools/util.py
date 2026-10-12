import functools
import json
import traceback
import types
from contextlib import contextmanager

import httpx
import langchain.tools
from ichatbio.agent_response import (
    ArtifactResponse,
    IChatBioAgentProcess,
    ResponseContext,
)
from ichatbio.types import Artifact

from context import current_context

JSON = dict | list | str | int | float | None
"""JSON-serializable primitive types that work with functions like json.dumps(). Note that dicts and lists may contain
content that is not JSON-serializable."""


def contains_non_null_content(content: JSON):
    """
    Returns True only if the JSON-serializable content contains a non-empty value. For example, returns True for [[1]]
    and False for [[]].
    """
    match content:
        case None:
            return False
        case list() as l:
            return any([contains_non_null_content(v) for v in l])
        case dict() as d:
            return any([contains_non_null_content(v) for k, v in d.items()])
        case _:
            return True


def context_tool(func):
    """
    Turns the function into a langchain tool that emits iChatBio messages.
    """

    @langchain.tools.tool(
        func.__name__,
        description=func.__doc__,
        return_direct=False,
    )
    @functools.wraps(func)  # Preserves function signature
    async def wrapper(*args, **kwargs):
        context = current_context.get()
        with capture_messages(context) as messages:
            result = await func(*args, **kwargs)
            if result is not None:
                return result
            return messages  # Pass the iChatBio messages back to the LangChain agent as context

    return wrapper


@contextmanager
def capture_messages(context: ResponseContext):
    """
    Modifies a ResponseContext so that any messages sent back to iChatBio are also collected into a list.

    Usage:
        context: ResponseContext
        with capture_messages(context) as messages:
            await context.reply("Alert!")
            # Now messages[0] is a DirectResponse object
    """
    messages = []

    channel = context._channel
    old_submit = channel.submit

    async def submit_and_buffer(self, message):
        await old_submit(message)
        match message:
            case ArtifactResponse() as artifact:
                # Remove "content" from artifact messages, the AI doesn't need to see it:
                # - The artifact description and metadata provide enough context for decision-making
                # - If the AI sees content, it may process it directly instead of running reliable processes
                # - It can be expensive to include content in LLM context, and it might not even fit
                messages.append(
                    ArtifactResponse(
                        description=artifact.description,
                        mimetype=artifact.mimetype,
                        metadata=artifact.metadata,
                    )
                )
            case _:
                messages.append(message)

    channel.submit = types.MethodType(submit_and_buffer, channel)

    yield messages

    channel.submit = old_submit


# JSON schema extraction



def format_exception(e) -> str:
    return "; ".join(traceback.format_exception(e, limit=0))


class ProcessError(Exception):
    """Something went wrong during an agent process."""


async def retrieve_text_artifact(
    artifact: Artifact, process: IChatBioAgentProcess
) -> str:
    """Retrieves artifact content as raw text (for CSV and text formats)."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as internet:
            for url in artifact.get_urls():
                await process.log(
                    f"Retrieving artifact {artifact.local_id} content from {url}"
                )
                response = await internet.get(url)
                if response.is_success:
                    return response.text
                else:
                    await process.log(
                        f"Error downloading artifact content: {response.reason_phrase} ({response.status_code})"
                    )
                    raise ProcessError()

            else:
                await process.log("Failed to find resolvable URL among artifact's URIs")
                raise ProcessError()
    except httpx.HTTPError as e:
        await process.log(f"Error retrieving artifact content: {format_exception(e)}")
        raise ProcessError() from e

