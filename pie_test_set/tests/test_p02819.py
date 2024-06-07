from pie_test_set.p02819 import problem_p02819


def test_problem_p02819_0():
    actual_output = problem_p02819("20")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02819_1():
    actual_output = problem_p02819("99992")
    expected_output = "100003"
    assert str(actual_output) == expected_output


def test_problem_p02819_2():
    actual_output = problem_p02819("20")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02819_3():
    actual_output = problem_p02819("2")
    expected_output = "2"
    assert str(actual_output) == expected_output
