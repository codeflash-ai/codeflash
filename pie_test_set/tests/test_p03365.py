from pie_test_set.p03365 import problem_p03365


def test_problem_p03365_0():
    actual_output = problem_p03365("4")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03365_1():
    actual_output = problem_p03365("2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03365_2():
    actual_output = problem_p03365("4")
    expected_output = "16"
    assert str(actual_output) == expected_output


def test_problem_p03365_3():
    actual_output = problem_p03365("5")
    expected_output = "84"
    assert str(actual_output) == expected_output


def test_problem_p03365_4():
    actual_output = problem_p03365("100000")
    expected_output = "341429644"
    assert str(actual_output) == expected_output
