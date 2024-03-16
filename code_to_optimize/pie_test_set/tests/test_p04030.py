from code_to_optimize.pie_test_set.p04030 import problem_p04030


def test_problem_p04030_0():
    actual_output = problem_p04030("01B0")
    expected_output = "00"
    assert str(actual_output) == expected_output


def test_problem_p04030_1():
    actual_output = problem_p04030("01B0")
    expected_output = "00"
    assert str(actual_output) == expected_output


def test_problem_p04030_2():
    actual_output = problem_p04030("0BB1")
    expected_output = "1"
    assert str(actual_output) == expected_output
