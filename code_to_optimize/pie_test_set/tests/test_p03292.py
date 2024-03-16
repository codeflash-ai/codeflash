from code_to_optimize.pie_test_set.p03292 import problem_p03292


def test_problem_p03292_0():
    actual_output = problem_p03292("1 6 3")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03292_1():
    actual_output = problem_p03292("1 6 3")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03292_2():
    actual_output = problem_p03292("100 100 100")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03292_3():
    actual_output = problem_p03292("11 5 5")
    expected_output = "6"
    assert str(actual_output) == expected_output
