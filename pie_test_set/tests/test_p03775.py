from pie_test_set.p03775 import problem_p03775


def test_problem_p03775_0():
    actual_output = problem_p03775("10000")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03775_1():
    actual_output = problem_p03775("1000003")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03775_2():
    actual_output = problem_p03775("10000")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03775_3():
    actual_output = problem_p03775("9876543210")
    expected_output = "6"
    assert str(actual_output) == expected_output
