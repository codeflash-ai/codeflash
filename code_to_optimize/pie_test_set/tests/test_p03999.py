from code_to_optimize.pie_test_set.p03999 import problem_p03999


def test_problem_p03999_0():
    actual_output = problem_p03999("125")
    expected_output = "176"
    assert str(actual_output) == expected_output


def test_problem_p03999_1():
    actual_output = problem_p03999("9999999999")
    expected_output = "12656242944"
    assert str(actual_output) == expected_output


def test_problem_p03999_2():
    actual_output = problem_p03999("125")
    expected_output = "176"
    assert str(actual_output) == expected_output
