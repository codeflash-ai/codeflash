from pie_test_set.p03643 import problem_p03643


def test_problem_p03643_0():
    actual_output = problem_p03643("100")
    expected_output = "ABC100"
    assert str(actual_output) == expected_output


def test_problem_p03643_1():
    actual_output = problem_p03643("999")
    expected_output = "ABC999"
    assert str(actual_output) == expected_output


def test_problem_p03643_2():
    actual_output = problem_p03643("425")
    expected_output = "ABC425"
    assert str(actual_output) == expected_output


def test_problem_p03643_3():
    actual_output = problem_p03643("100")
    expected_output = "ABC100"
    assert str(actual_output) == expected_output
