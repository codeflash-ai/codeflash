from pie_test_set.p03079 import problem_p03079


def test_problem_p03079_0():
    actual_output = problem_p03079("2 2 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03079_1():
    actual_output = problem_p03079("2 2 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03079_2():
    actual_output = problem_p03079("3 4 5")
    expected_output = "No"
    assert str(actual_output) == expected_output
