from code_to_optimize.pie_test_set.p03105 import problem_p03105


def test_problem_p03105_0():
    actual_output = problem_p03105("2 11 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03105_1():
    actual_output = problem_p03105("3 9 5")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03105_2():
    actual_output = problem_p03105("2 11 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03105_3():
    actual_output = problem_p03105("100 1 10")
    expected_output = "0"
    assert str(actual_output) == expected_output
