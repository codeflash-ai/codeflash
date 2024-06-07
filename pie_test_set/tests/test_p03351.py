from pie_test_set.p03351 import problem_p03351


def test_problem_p03351_0():
    actual_output = problem_p03351("4 7 9 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03351_1():
    actual_output = problem_p03351("1 100 2 10")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03351_2():
    actual_output = problem_p03351("100 10 1 2")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03351_3():
    actual_output = problem_p03351("10 10 10 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03351_4():
    actual_output = problem_p03351("4 7 9 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
