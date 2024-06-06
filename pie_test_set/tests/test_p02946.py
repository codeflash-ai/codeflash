from pie_test_set.p02946 import problem_p02946


def test_problem_p02946_0():
    actual_output = problem_p02946("3 7")
    expected_output = "5 6 7 8 9"
    assert str(actual_output) == expected_output


def test_problem_p02946_1():
    actual_output = problem_p02946("4 0")
    expected_output = "-3 -2 -1 0 1 2 3"
    assert str(actual_output) == expected_output


def test_problem_p02946_2():
    actual_output = problem_p02946("1 100")
    expected_output = "100"
    assert str(actual_output) == expected_output


def test_problem_p02946_3():
    actual_output = problem_p02946("3 7")
    expected_output = "5 6 7 8 9"
    assert str(actual_output) == expected_output
