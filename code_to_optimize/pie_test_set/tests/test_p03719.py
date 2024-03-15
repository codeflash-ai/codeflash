from code_to_optimize.pie_test_set.p03719 import problem_p03719


def test_problem_p03719_0():
    actual_output = problem_p03719("1 3 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03719_1():
    actual_output = problem_p03719("6 5 4")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03719_2():
    actual_output = problem_p03719("2 2 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03719_3():
    actual_output = problem_p03719("1 3 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
