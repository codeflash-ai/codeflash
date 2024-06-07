from pie_test_set.p02755 import problem_p02755


def test_problem_p02755_0():
    actual_output = problem_p02755("2 2")
    expected_output = "25"
    assert str(actual_output) == expected_output


def test_problem_p02755_1():
    actual_output = problem_p02755("19 99")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p02755_2():
    actual_output = problem_p02755("2 2")
    expected_output = "25"
    assert str(actual_output) == expected_output


def test_problem_p02755_3():
    actual_output = problem_p02755("8 10")
    expected_output = "100"
    assert str(actual_output) == expected_output
