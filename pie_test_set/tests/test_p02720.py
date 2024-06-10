from pie_test_set.p02720 import problem_p02720


def test_problem_p02720_0():
    actual_output = problem_p02720("15")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02720_1():
    actual_output = problem_p02720("13")
    expected_output = "21"
    assert str(actual_output) == expected_output


def test_problem_p02720_2():
    actual_output = problem_p02720("1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02720_3():
    actual_output = problem_p02720("15")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02720_4():
    actual_output = problem_p02720("100000")
    expected_output = "3234566667"
    assert str(actual_output) == expected_output
