from pie_test_set.p02467 import problem_p02467


def test_problem_p02467_0():
    actual_output = problem_p02467("12")
    expected_output = "12: 2 2 3"
    assert str(actual_output) == expected_output


def test_problem_p02467_1():
    actual_output = problem_p02467("12")
    expected_output = "12: 2 2 3"
    assert str(actual_output) == expected_output


def test_problem_p02467_2():
    actual_output = problem_p02467("126")
    expected_output = "126: 2 3 3 7"
    assert str(actual_output) == expected_output
