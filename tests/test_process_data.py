import json

import dotenv
import ichatbio.agent_response
import pytest
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

from artifact_registry import ArtifactRegistry
from conftest import resource
from context import (
    current_request,
    current_context,
    current_artifacts,
    ValidatedArtifactID,
)
from tools import process_data

dotenv.load_dotenv()

OCCURRENCE_RECORDS = Artifact(
    local_id="#0000",
    mimetype="application/json",
    description="A list of occurrence records",
    uris=["https://artifact.test"],
    metadata={"source": "iDigBio"},
)


@pytest.mark.httpx_mock(
    should_mock=lambda request: request.url == "https://artifact.test"
)
class TestWithArtifactAccess:

    @pytest.fixture(autouse=True)
    def setup(self, httpx_mock, context):
        self.artifact_content = json.loads(
            resource("idigbio_records_search_result.json")
        )
        httpx_mock.add_response(url="https://artifact.test", json=self.artifact_content)

        current_context.set(context)
        current_artifacts.set(
            ArtifactRegistry(
                [
                    Artifact(
                        local_id="#0000",
                        mimetype="application/json",
                        description="A list of occurrence records",
                        uris=["https://artifact.test"],
                        metadata={"source": "iDigBio"},
                    )
                ]
            )
        )

    async def run_tool(self, request: str, artifact_id: ValidatedArtifactID):
        current_request.set(request)
        await process_data.process_data.ainvoke({"artifact_id": artifact_id})

    @pytest.mark.asyncio
    async def test_extract_first_record(self, messages):
        await self.run_tool("Get the first record", "#0000")

        artifact_message = next(
            (m for m in messages if isinstance(m, ArtifactResponse))
        )
        assert artifact_message

        first_record = json.loads(artifact_message.content.decode("utf-8"))

        # If the record was put into a single-item list, pull it out
        if type(first_record) is list:
            assert len(first_record) == 1
            first_record = first_record[0]

        assert first_record == self.artifact_content["items"][0]

    @pytest.mark.httpx_mock(
        should_mock=lambda request: request.url == "https://artifact.test"
    )
    @pytest.mark.asyncio
    async def test_abort_after_failed_query(self, messages):
        await self.run_tool('Extract "dwc.moonphases" from these records', "#0000")

        assert len(messages) > 1
        assert isinstance(messages[0], ichatbio.agent_response.ProcessBeginResponse)
        assert not any([isinstance(message, ArtifactResponse) for message in messages])

    @pytest.mark.asyncio
    async def test_extract_a_list(self, messages):
        await self.run_tool("Extract records list", "#0000")

        artifact_message = next(
            (m for m in messages if isinstance(m, ArtifactResponse))
        )
        assert artifact_message

        records = json.loads(artifact_message.content.decode("utf-8"))

        assert records == self.artifact_content["items"]
