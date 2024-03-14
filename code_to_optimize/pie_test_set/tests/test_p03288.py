from code_to_optimize.pie_test_set.p03288 import problem_p03288


def test_problem_p03288_0():
    actual_output = problem_p03288("1199")
    expected_output = "ABC"
    assert str(actual_output) == expected_output


def test_problem_p03288_1():
    actual_output = problem_p03288("1199")
    expected_output = "ABC"
    assert str(actual_output) == expected_output


def test_problem_p03288_2():
    actual_output = problem_p03288("1200")
    expected_output = "ARC"
    assert str(actual_output) == expected_output


def test_problem_p03288_3():
    actual_output = problem_p03288("4208")
    expected_output = "AGC"
    assert str(actual_output) == expected_output
