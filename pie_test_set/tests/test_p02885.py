from pie_test_set.p02885 import problem_p02885


def test_problem_p02885_0():
    actual_output = problem_p02885("12 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02885_1():
    actual_output = problem_p02885("20 30")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02885_2():
    actual_output = problem_p02885("12 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02885_3():
    actual_output = problem_p02885("20 15")
    expected_output = "0"
    assert str(actual_output) == expected_output
