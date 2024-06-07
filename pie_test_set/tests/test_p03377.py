from pie_test_set.p03377 import problem_p03377


def test_problem_p03377_0():
    actual_output = problem_p03377("3 5 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03377_1():
    actual_output = problem_p03377("2 2 6")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03377_2():
    actual_output = problem_p03377("3 5 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03377_3():
    actual_output = problem_p03377("5 3 2")
    expected_output = "NO"
    assert str(actual_output) == expected_output
