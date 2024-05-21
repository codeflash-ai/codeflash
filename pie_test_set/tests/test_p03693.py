from pie_test_set.p03693 import problem_p03693


def test_problem_p03693_0():
    actual_output = problem_p03693("4 3 2")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03693_1():
    actual_output = problem_p03693("4 3 2")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03693_2():
    actual_output = problem_p03693("2 3 4")
    expected_output = "NO"
    assert str(actual_output) == expected_output
