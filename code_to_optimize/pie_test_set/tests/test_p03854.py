from code_to_optimize.pie_test_set.p03854 import problem_p03854


def test_problem_p03854_0():
    actual_output = problem_p03854("erasedream")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03854_1():
    actual_output = problem_p03854("dreamerer")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03854_2():
    actual_output = problem_p03854("erasedream")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03854_3():
    actual_output = problem_p03854("dreameraser")
    expected_output = "YES"
    assert str(actual_output) == expected_output
