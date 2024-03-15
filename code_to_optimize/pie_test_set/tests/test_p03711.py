from code_to_optimize.pie_test_set.p03711 import problem_p03711


def test_problem_p03711_0():
    actual_output = problem_p03711("1 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03711_1():
    actual_output = problem_p03711("2 4")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03711_2():
    actual_output = problem_p03711("1 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
