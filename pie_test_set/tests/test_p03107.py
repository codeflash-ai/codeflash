from pie_test_set.p03107 import problem_p03107


def test_problem_p03107_0():
    actual_output = problem_p03107("0011")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03107_1():
    actual_output = problem_p03107("0")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03107_2():
    actual_output = problem_p03107("0011")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03107_3():
    actual_output = problem_p03107("11011010001011")
    expected_output = "12"
    assert str(actual_output) == expected_output
