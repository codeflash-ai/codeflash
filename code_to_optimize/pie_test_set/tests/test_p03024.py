from code_to_optimize.pie_test_set.p03024 import problem_p03024


def test_problem_p03024_0():
    actual_output = problem_p03024("oxoxoxoxoxoxox")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03024_1():
    actual_output = problem_p03024("oxoxoxoxoxoxox")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03024_2():
    actual_output = problem_p03024("xxxxxxxx")
    expected_output = "NO"
    assert str(actual_output) == expected_output
