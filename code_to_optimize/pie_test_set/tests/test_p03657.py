from code_to_optimize.pie_test_set.p03657 import problem_p03657


def test_problem_p03657_0():
    actual_output = problem_p03657("4 5")
    expected_output = "Possible"
    assert str(actual_output) == expected_output


def test_problem_p03657_1():
    actual_output = problem_p03657("4 5")
    expected_output = "Possible"
    assert str(actual_output) == expected_output


def test_problem_p03657_2():
    actual_output = problem_p03657("1 1")
    expected_output = "Impossible"
    assert str(actual_output) == expected_output
