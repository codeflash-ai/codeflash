from pie_test_set.p03332 import problem_p03332


def test_problem_p03332_0():
    actual_output = problem_p03332("4 1 2 5")
    expected_output = "40"
    assert str(actual_output) == expected_output


def test_problem_p03332_1():
    actual_output = problem_p03332("90081 33447 90629 6391049189")
    expected_output = "577742975"
    assert str(actual_output) == expected_output


def test_problem_p03332_2():
    actual_output = problem_p03332("2 5 6 0")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03332_3():
    actual_output = problem_p03332("4 1 2 5")
    expected_output = "40"
    assert str(actual_output) == expected_output
