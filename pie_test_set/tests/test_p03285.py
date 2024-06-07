from pie_test_set.p03285 import problem_p03285


def test_problem_p03285_0():
    actual_output = problem_p03285("11")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03285_1():
    actual_output = problem_p03285("40")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03285_2():
    actual_output = problem_p03285("3")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03285_3():
    actual_output = problem_p03285("11")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
