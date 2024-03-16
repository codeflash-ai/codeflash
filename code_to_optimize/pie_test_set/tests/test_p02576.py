from code_to_optimize.pie_test_set.p02576 import problem_p02576


def test_problem_p02576_0():
    actual_output = problem_p02576("20 12 6")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p02576_1():
    actual_output = problem_p02576("1000 1 1000")
    expected_output = "1000000"
    assert str(actual_output) == expected_output


def test_problem_p02576_2():
    actual_output = problem_p02576("20 12 6")
    expected_output = "12"
    assert str(actual_output) == expected_output
