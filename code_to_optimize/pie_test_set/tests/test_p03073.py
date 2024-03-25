from code_to_optimize.pie_test_set.p03073 import problem_p03073


def test_problem_p03073_0():
    actual_output = problem_p03073("000")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03073_1():
    actual_output = problem_p03073("10010010")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03073_2():
    actual_output = problem_p03073("0")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03073_3():
    actual_output = problem_p03073("000")
    expected_output = "1"
    assert str(actual_output) == expected_output
