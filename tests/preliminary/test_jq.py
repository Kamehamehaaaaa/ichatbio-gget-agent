import jq


def test_jq():
    assert jq.compile(".").input_value("1\n2\n3").first() == "1\n2\n3"

    assert jq.compile(".").input_value([1, 2, 3]).first() == [1, 2, 3]

    assert jq.compile(".").input_value([1, 2, 3]).all() == [[1, 2, 3]]

    assert jq.compile(".[]+1").input_value([1, 2, 3]).all() == [2, 3, 4]

    assert jq.compile(".[]+1").input_value([1, 2, 3]).first() == 2

    assert jq.compile(".items").input_value({"items": [1, 2, 3]}).first() == [1, 2, 3]

    assert jq.compile(".[].items").input_value([{"items": [1, 2, 3]}]).first() == [
        1,
        2,
        3,
    ]
