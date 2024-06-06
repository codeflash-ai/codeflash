from pie_test_set.p03502 import problem_p03502


def test_problem_p03502_0():
    actual_output = problem_p03502("12")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p03502_1():
    actual_output = problem_p03502("148")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03502_2():
    actual_output = problem_p03502("57")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p03502_3():
    actual_output = problem_p03502("12")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
