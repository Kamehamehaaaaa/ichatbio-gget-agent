from contextvars import ContextVar
from typing import Annotated

from ichatbio.agent_response import ResponseContext
from pydantic import AfterValidator, Field

from artifact_registry import ArtifactID, ArtifactRegistry

current_request: ContextVar[str] = ContextVar("current_request")
current_context: ContextVar[ResponseContext] = ContextVar("current_context")
current_artifacts: ContextVar[ArtifactRegistry] = ContextVar("current_artifacts")


def check_artifact_exists(local_id: str):
    artifacts = current_artifacts.get()
    artifacts.model = ValidatedArtifactID
    artifacts.get(ArtifactID(local_id))  # Raises an exception if artifact doesn't exist
    return local_id


ValidatedArtifactID = Annotated[
    str,
    Field(pattern="^#[0-9a-f]{4}$", examples=["#01ef"]),
    AfterValidator(check_artifact_exists),
]
