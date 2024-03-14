from code_to_optimize.pie_test_set.p02847 import problem_p02847


def test_problem_p02847_0():
    actual_output = problem_p02847("SAT")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02847_1():
    actual_output = problem_p02847("SUN")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02847_2():
    actual_output = problem_p02847("SAT")
    expected_output = "1"
    assert str(actual_output) == expected_output
