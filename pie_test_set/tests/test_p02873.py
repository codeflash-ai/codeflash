from pie_test_set.p02873 import problem_p02873


def test_problem_p02873_0():
    actual_output = problem_p02873("<>>")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02873_1():
    actual_output = problem_p02873("<>>><<><<<<<>>><")
    expected_output = "28"
    assert str(actual_output) == expected_output


def test_problem_p02873_2():
    actual_output = problem_p02873("<>>")
    expected_output = "3"
    assert str(actual_output) == expected_output
