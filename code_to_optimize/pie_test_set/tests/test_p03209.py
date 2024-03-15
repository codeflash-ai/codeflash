from code_to_optimize.pie_test_set.p03209 import problem_p03209


def test_problem_p03209_0():
    actual_output = problem_p03209("2 7")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03209_1():
    actual_output = problem_p03209("50 4321098765432109")
    expected_output = "2160549382716056"
    assert str(actual_output) == expected_output


def test_problem_p03209_2():
    actual_output = problem_p03209("1 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03209_3():
    actual_output = problem_p03209("2 7")
    expected_output = "4"
    assert str(actual_output) == expected_output
