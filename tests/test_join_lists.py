import json

import pytest
from ichatbio.agent_response import (
    ArtifactResponse,
)
from ichatbio.types import Artifact

from artifact_registry import ArtifactRegistry
from conftest import resource
from context import current_context, current_artifacts, ValidatedArtifactID
from tools import join_lists


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
        await join_lists.join_lists.ainvoke(
            {"artifact_one_id": artifact_one_id, "artifact_two_id": artifact_two_id}
        )

    @pytest.mark.httpx_mock(
        should_mock=lambda request: request.url
        in ("https://artifact.test/list_one", "https://artifact.test/list_two")
    )
    @pytest.mark.asyncio
    async def test_join_lists_by_index(self, messages, httpx_mock):
        list_one = json.loads(resource("list_of_idigbio_records.json"))
        httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

        list_two = [{"new_field": "fee"}, {"new_field": "fi"}, {"new_field": "fo"}]
        httpx_mock.add_response(url="https://artifact.test/list_two", json=list_two)

        expected_list = [
            record_one | record_two
            for record_one, record_two in zip(list_one, list_two)
        ]

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
    async def test_dont_join_things_that_are_not_lists(self, messages, httpx_mock):
        list_one = json.loads('"this is not a list"')
        httpx_mock.add_response(url="https://artifact.test/list_one", json=list_one)

        await self.run_tool("#1111", "#2222")

        artifact_message = next(
            (m for m in messages if isinstance(m, ArtifactResponse)), None
        )
        assert artifact_message is None
