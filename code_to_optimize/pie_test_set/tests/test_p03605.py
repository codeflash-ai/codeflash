from code_to_optimize.pie_test_set.p03605 import problem_p03605


def test_problem_p03605_0():
    actual_output = problem_p03605("29")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03605_1():
    actual_output = problem_p03605("29")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03605_2():
    actual_output = problem_p03605("91")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03605_3():
    actual_output = problem_p03605("72")
    expected_output = "No"
    assert str(actual_output) == expected_output
