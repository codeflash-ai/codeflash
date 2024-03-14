from code_to_optimize.pie_test_set.p03369 import problem_p03369


def test_problem_p03369_0():
    actual_output = problem_p03369("oxo")
    expected_output = "900"
    assert str(actual_output) == expected_output


def test_problem_p03369_1():
    actual_output = problem_p03369("oxo")
    expected_output = "900"
    assert str(actual_output) == expected_output


def test_problem_p03369_2():
    actual_output = problem_p03369("xxx")
    expected_output = "700"
    assert str(actual_output) == expected_output


def test_problem_p03369_3():
    actual_output = problem_p03369("ooo")
    expected_output = "1000"
    assert str(actual_output) == expected_output
