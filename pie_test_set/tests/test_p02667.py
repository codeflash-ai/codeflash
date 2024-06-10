from pie_test_set.p02667 import problem_p02667


def test_problem_p02667_0():
    actual_output = problem_p02667("1101")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p02667_1():
    actual_output = problem_p02667("1101")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p02667_2():
    actual_output = problem_p02667("0111101101")
    expected_output = "26"
    assert str(actual_output) == expected_output
