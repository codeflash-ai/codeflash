from pie_test_set.p03683 import problem_p03683


def test_problem_p03683_0():
    actual_output = problem_p03683("2 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03683_1():
    actual_output = problem_p03683("100000 100000")
    expected_output = "530123477"
    assert str(actual_output) == expected_output


def test_problem_p03683_2():
    actual_output = problem_p03683("3 2")
    expected_output = "12"
    assert str(actual_output) == expected_output


def test_problem_p03683_3():
    actual_output = problem_p03683("1 8")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03683_4():
    actual_output = problem_p03683("2 2")
    expected_output = "8"
    assert str(actual_output) == expected_output
