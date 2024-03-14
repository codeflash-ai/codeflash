from code_to_optimize.pie_test_set.p03129 import problem_p03129


def test_problem_p03129_0():
    actual_output = problem_p03129("3 2")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03129_1():
    actual_output = problem_p03129("10 90")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03129_2():
    actual_output = problem_p03129("31 10")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03129_3():
    actual_output = problem_p03129("3 2")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03129_4():
    actual_output = problem_p03129("5 5")
    expected_output = "NO"
    assert str(actual_output) == expected_output
