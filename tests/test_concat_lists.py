import json

import pytest
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

from artifact_registry import ArtifactRegistry
from conftest import resource
from context import current_context, current_artifacts, ValidatedArtifactID
from tools import concat_lists


class TestWithArtifactAccess:

    @pytest.fixture(autouse=True)
    def setup(self, context):
        current_context.set(context)
        current_artifacts.set(
            ArtifactRegistry(
                [
                    Artifact(
                        local_id="#1111",
                        mimetype="application/json",
                        description="A list of occurrence records",
                        uris=["https://artifact.test/list_one"],
                        metadata={},
                    ),
                    Artifact(
                        local_id="#2222",
                        mimetype="application/json",
                        description="A list of extra data",
                        uris=["https://artifact.test/list_two"],
                        metadata={},
                    ),
                ]
            )
        )

    async def run_tool(
        self, artifact_one_id: ValidatedArtifactID, artifact_two_id: ValidatedArtifactID
    ):
        await concat_lists.concat_lists.ainvoke(
            {"artifact_one_id": artifact_one_id, "artifact_two_id": artifact_two_id}
        )

    @pytest.mark.httpx_mock(
        should_mock=lambda request: request.url
        in ("https://artifact.test/list_one", "https://artifact.test/list_two")
    )
    @pytest.mark.asyncio
    async def test_concat_lists(self, messages, httpx_mock):
        source_list = json.loads(resource("list_of_idigbio_records.json"))

        list_one = source_list[:1]
        httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

        list_two = source_list[2:]
        httpx_mock.add_response(url="https://artifact.test/list_two", json=list_two)

        expected_list = list_one + list_two

        await self.run_tool("#1111", "#2222")

        artifact_message = next(
            (m for m in messages if isinstance(m, ArtifactResponse)), None
        )
        assert artifact_message

        new_list = json.loads(artifact_message.content.decode("utf-8"))

        assert new_list == expected_list

    @pytest.mark.httpx_mock(
        should_mock=lambda request: request.url in ("https://artifact.test/list_one",)
    )
    @pytest.mark.asyncio
    async def test_dont_concat_things_that_are_not_lists(self, messages, httpx_mock):
        list_one = json.loads('"this is not a list"')
        httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

        await self.run_tool("#1111", "#2222")

        artifact_message = next(
            (m for m in messages if isinstance(m, ArtifactResponse)), None
        )
        assert artifact_message is None
