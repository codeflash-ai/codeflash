from code_to_optimize.final_test_set.single_name_to_first_last_names import (
    single_name_to_first_last_names,
)


def test_two_part_name():
    name = "John Doe"
    expected = [("JOHN", "DOE")]
    result = single_name_to_first_last_names(name)
    assert result == expected


def test_three_part_name():
    name = "John Michael Doe"
    expected = [("JOHN", "DOE"), ("JOHN", "MICHAEL DOE"), ("JOHN MICHAEL", "DOE")]
    result = single_name_to_first_last_names(name)
    assert result == expected


def test_single_part_name():
    name = "Prince"
    expected = []
    result = single_name_to_first_last_names(name)
    assert result == expected


def test_more_than_three_parts():
    name = "John Michael Andrew Doe"
    expected = []
    result = single_name_to_first_last_names(name)
    assert result == expected


def test_empty_string():
    name = ""
    expected = []
    result = single_name_to_first_last_names(name)
    assert result == expected
