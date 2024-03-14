from code_to_optimize.pie_test_set.p03139 import problem_p03139


def test_problem_p03139_0():
    actual_output = problem_p03139("10 3 5")
    expected_output = "3 0"
    assert str(actual_output) == expected_output


def test_problem_p03139_1():
    actual_output = problem_p03139("10 7 5")
    expected_output = "5 2"
    assert str(actual_output) == expected_output


def test_problem_p03139_2():
    actual_output = problem_p03139("100 100 100")
    expected_output = "100 100"
    assert str(actual_output) == expected_output


def test_problem_p03139_3():
    actual_output = problem_p03139("10 3 5")
    expected_output = "3 0"
    assert str(actual_output) == expected_output
