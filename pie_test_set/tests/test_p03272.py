from pie_test_set.p03272 import problem_p03272


def test_problem_p03272_0():
    actual_output = problem_p03272("4 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03272_1():
    actual_output = problem_p03272("1 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03272_2():
    actual_output = problem_p03272("4 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03272_3():
    actual_output = problem_p03272("15 11")
    expected_output = "5"
    assert str(actual_output) == expected_output
