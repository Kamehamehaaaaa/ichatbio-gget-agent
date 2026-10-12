import importlib.resources
import json

from tools.util import contains_non_null_content, extract_json_schema


class TestContainsNonNullContent:
    def test_no_content_is_null(self):
        assert not contains_non_null_content(None)

    def test_empty_list_is_null(self):
        assert not contains_non_null_content([])

    def test_empty_dict_is_null(self):
        assert not contains_non_null_content({})

    def test_empty_leaves_is_null(self):
        assert not contains_non_null_content([{}, []])

    def test_primitive_is_non_null(self):
        assert contains_non_null_content("haha")

    def test_empty_string_is_non_null(self):
        assert contains_non_null_content("")

    def test_partially_null_is_non_null(self):
        assert contains_non_null_content(["haha", None])

    def test_nested_is_non_null(self):
        assert contains_non_null_content({"key": {"key": "value"}})


def test_extract_json_schema():
    data = json.loads(
        importlib.resources.files("resources")
        .joinpath("idigbio_records_search_result.json")
        .read_text()
    )
    schema = extract_json_schema(data)
    top_level_fields = set(schema["properties"]["items"]["items"]["properties"].keys())
    assert top_level_fields == {"data", "etag", "indexTerms", "type", "uuid"}
