from pie_test_set.p03773 import problem_p03773


def test_problem_p03773_0():
    actual_output = problem_p03773("9 12")
    expected_output = "21"
    assert str(actual_output) == expected_output


def test_problem_p03773_1():
    actual_output = problem_p03773("9 12")
    expected_output = "21"
    assert str(actual_output) == expected_output


def test_problem_p03773_2():
    actual_output = problem_p03773("19 0")
    expected_output = "19"
    assert str(actual_output) == expected_output


def test_problem_p03773_3():
    actual_output = problem_p03773("23 2")
    expected_output = "1"
    assert str(actual_output) == expected_output
