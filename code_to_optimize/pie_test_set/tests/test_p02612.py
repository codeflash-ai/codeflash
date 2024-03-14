from code_to_optimize.pie_test_set.p02612 import problem_p02612


def test_problem_p02612_0():
    actual_output = problem_p02612("1900")
    expected_output = "100"
    assert str(actual_output) == expected_output


def test_problem_p02612_1():
    actual_output = problem_p02612("3000")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02612_2():
    actual_output = problem_p02612("1900")
    expected_output = "100"
    assert str(actual_output) == expected_output
