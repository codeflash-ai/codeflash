from code_to_optimize.pie_test_set.p03088 import problem_p03088


def test_problem_p03088_0():
    actual_output = problem_p03088("3")
    expected_output = "61"
    assert str(actual_output) == expected_output


def test_problem_p03088_1():
    actual_output = problem_p03088("4")
    expected_output = "230"
    assert str(actual_output) == expected_output


def test_problem_p03088_2():
    actual_output = problem_p03088("3")
    expected_output = "61"
    assert str(actual_output) == expected_output


def test_problem_p03088_3():
    actual_output = problem_p03088("100")
    expected_output = "388130742"
    assert str(actual_output) == expected_output
