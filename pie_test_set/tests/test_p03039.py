from pie_test_set.p03039 import problem_p03039


def test_problem_p03039_0():
    actual_output = problem_p03039("2 2 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03039_1():
    actual_output = problem_p03039("2 2 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03039_2():
    actual_output = problem_p03039("100 100 5000")
    expected_output = "817260251"
    assert str(actual_output) == expected_output


def test_problem_p03039_3():
    actual_output = problem_p03039("4 5 4")
    expected_output = "87210"
    assert str(actual_output) == expected_output
