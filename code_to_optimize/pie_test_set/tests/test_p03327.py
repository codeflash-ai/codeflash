from code_to_optimize.pie_test_set.p03327 import problem_p03327


def test_problem_p03327_0():
    actual_output = problem_p03327("999")
    expected_output = "ABC"
    assert str(actual_output) == expected_output


def test_problem_p03327_1():
    actual_output = problem_p03327("1000")
    expected_output = "ABD"
    assert str(actual_output) == expected_output


def test_problem_p03327_2():
    actual_output = problem_p03327("1481")
    expected_output = "ABD"
    assert str(actual_output) == expected_output


def test_problem_p03327_3():
    actual_output = problem_p03327("999")
    expected_output = "ABC"
    assert str(actual_output) == expected_output
