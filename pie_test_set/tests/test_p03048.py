from pie_test_set.p03048 import problem_p03048


def test_problem_p03048_0():
    actual_output = problem_p03048("1 2 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03048_1():
    actual_output = problem_p03048("1 2 3 4")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03048_2():
    actual_output = problem_p03048("13 1 4 3000")
    expected_output = "87058"
    assert str(actual_output) == expected_output
