from code_to_optimize.pie_test_set.p02922 import problem_p02922


def test_problem_p02922_0():
    actual_output = problem_p02922("4 10")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02922_1():
    actual_output = problem_p02922("4 10")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02922_2():
    actual_output = problem_p02922("8 8")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02922_3():
    actual_output = problem_p02922("8 9")
    expected_output = "2"
    assert str(actual_output) == expected_output
