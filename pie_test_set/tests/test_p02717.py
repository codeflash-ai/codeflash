from pie_test_set.p02717 import problem_p02717


def test_problem_p02717_0():
    actual_output = problem_p02717("1 2 3")
    expected_output = "3 1 2"
    assert str(actual_output) == expected_output


def test_problem_p02717_1():
    actual_output = problem_p02717("100 100 100")
    expected_output = "100 100 100"
    assert str(actual_output) == expected_output


def test_problem_p02717_2():
    actual_output = problem_p02717("41 59 31")
    expected_output = "31 41 59"
    assert str(actual_output) == expected_output


def test_problem_p02717_3():
    actual_output = problem_p02717("1 2 3")
    expected_output = "3 1 2"
    assert str(actual_output) == expected_output
