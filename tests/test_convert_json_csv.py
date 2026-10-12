import csv
import json
from io import StringIO

import pytest

from tools.convert_json_csv import _flatten_dict, _json_to_csv, _csv_to_json


class TestFlattenDict:
    def test_simple_dict(self):
        data = {"name": "John", "age": 30}
        result = _flatten_dict(data)
        assert result == {"name": "John", "age": 30}

    def test_nested_dict(self):
        data = {"person": {"name": "John", "age": 30}}
        result = _flatten_dict(data)
        assert result == {"person.name": "John", "person.age": 30}

    def test_dict_with_list(self):
        data = {"items": [1, 2, 3]}
        result = _flatten_dict(data)
        assert result == {"items[0]": 1, "items[1]": 2, "items[2]": 3}

    def test_dict_with_nested_list_of_dicts(self):
        data = {"people": [{"name": "John"}, {"name": "Jane"}]}
        result = _flatten_dict(data)
        assert result == {"people[0].name": "John", "people[1].name": "Jane"}


class TestJsonToCsv:
    def test_single_object(self):
        data = {"name": "John", "age": 30}
        result = _json_to_csv(data)
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "John"
        assert rows[0]["age"] == "30"

    def test_array_of_objects(self):
        data = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ]
        result = _json_to_csv(data)
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["name"] == "John"
        assert rows[1]["name"] == "Jane"

    def test_json_string_input(self):
        data_str = '{"name": "John", "age": 30}'
        result = _json_to_csv(data_str)
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "John"

    def test_nested_json(self):
        data = [
            {"person": {"name": "John", "age": 30}},
            {"person": {"name": "Jane", "age": 25}},
        ]
        result = _json_to_csv(data)
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["person.name"] == "John"
        assert rows[1]["person.name"] == "Jane"

    def test_invalid_json_string(self):
        data_str = "not valid json"
        with pytest.raises(ValueError, match="Invalid JSON input"):
            _json_to_csv(data_str)

    def test_invalid_json_type(self):
        data = '"just a string"'
        with pytest.raises(ValueError, match="JSON must be an object or array"):
            _json_to_csv(data)

    def test_empty_array(self):
        data = []
        result = _json_to_csv(data)
        assert result == ""


class TestCsvToJson:
    def test_simple_csv(self):
        csv_data = "name,age\nJohn,30\nJane,25"
        result = _csv_to_json(csv_data)
        records = json.loads(result)
        assert len(records) == 2
        assert records[0]["name"] == "John"
        assert records[0]["age"] == "30"
        assert records[1]["name"] == "Jane"
        assert records[1]["age"] == "25"

    def test_empty_csv(self):
        csv_data = ""
        result = _csv_to_json(csv_data)
        records = json.loads(result)
        assert records == []

    def test_whitespace_only_csv(self):
        csv_data = "   \n   "
        result = _csv_to_json(csv_data)
        records = json.loads(result)
        assert records == []


class TestRoundTrip:
    def test_json_to_csv_to_json(self):
        original = [
            {"name": "John", "age": 30},
            {"name": "Jane", "age": 25},
        ]
        csv_result = _json_to_csv(original)
        json_result = _csv_to_json(csv_result)
        records = json.loads(json_result)

        assert len(records) == 2
        assert records[0]["name"] == "John"
        assert records[0]["age"] == "30"  # Note: converted to string in CSV
        assert records[1]["name"] == "Jane"
