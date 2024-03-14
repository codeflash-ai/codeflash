from code_to_optimize.pie_test_set.p03483 import problem_p03483


def test_problem_p03483_0():
    actual_output = problem_p03483("eel")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03483_1():
    actual_output = problem_p03483("snuke")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03483_2():
    actual_output = problem_p03483("ataatmma")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03483_3():
    actual_output = problem_p03483("eel")
    expected_output = "1"
    assert str(actual_output) == expected_output
