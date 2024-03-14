from code_to_optimize.pie_test_set.p03067 import problem_p03067


def test_problem_p03067_0():
    actual_output = problem_p03067("3 8 5")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03067_1():
    actual_output = problem_p03067("10 2 4")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03067_2():
    actual_output = problem_p03067("7 3 1")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03067_3():
    actual_output = problem_p03067("31 41 59")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03067_4():
    actual_output = problem_p03067("3 8 5")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
