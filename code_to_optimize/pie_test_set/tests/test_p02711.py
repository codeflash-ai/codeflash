from code_to_optimize.pie_test_set.p02711 import problem_p02711


def test_problem_p02711_0():
    actual_output = problem_p02711("117")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02711_1():
    actual_output = problem_p02711("777")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02711_2():
    actual_output = problem_p02711("123")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02711_3():
    actual_output = problem_p02711("117")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
