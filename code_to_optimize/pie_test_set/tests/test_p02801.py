from code_to_optimize.pie_test_set.p02801 import problem_p02801


def test_problem_p02801_0():
    actual_output = problem_p02801("a")
    expected_output = "b"
    assert str(actual_output) == expected_output


def test_problem_p02801_1():
    actual_output = problem_p02801("y")
    expected_output = "z"
    assert str(actual_output) == expected_output


def test_problem_p02801_2():
    actual_output = problem_p02801("a")
    expected_output = "b"
    assert str(actual_output) == expected_output
