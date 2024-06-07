from pie_test_set.p03243 import problem_p03243


def test_problem_p03243_0():
    actual_output = problem_p03243("111")
    expected_output = "111"
    assert str(actual_output) == expected_output


def test_problem_p03243_1():
    actual_output = problem_p03243("750")
    expected_output = "777"
    assert str(actual_output) == expected_output


def test_problem_p03243_2():
    actual_output = problem_p03243("112")
    expected_output = "222"
    assert str(actual_output) == expected_output


def test_problem_p03243_3():
    actual_output = problem_p03243("111")
    expected_output = "111"
    assert str(actual_output) == expected_output
