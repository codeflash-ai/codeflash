from code_to_optimize.pie_test_set.p03558 import problem_p03558


def test_problem_p03558_0():
    actual_output = problem_p03558("6")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03558_1():
    actual_output = problem_p03558("41")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03558_2():
    actual_output = problem_p03558("6")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03558_3():
    actual_output = problem_p03558("79992")
    expected_output = "36"
    assert str(actual_output) == expected_output
