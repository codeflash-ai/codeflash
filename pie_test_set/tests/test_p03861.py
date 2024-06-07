from pie_test_set.p03861 import problem_p03861


def test_problem_p03861_0():
    actual_output = problem_p03861("4 8 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03861_1():
    actual_output = problem_p03861("4 8 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03861_2():
    actual_output = problem_p03861("9 9 2")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03861_3():
    actual_output = problem_p03861("1 1000000000000000000 3")
    expected_output = "333333333333333333"
    assert str(actual_output) == expected_output


def test_problem_p03861_4():
    actual_output = problem_p03861("0 5 1")
    expected_output = "6"
    assert str(actual_output) == expected_output
