from code_to_optimize.pie_test_set.p02981 import problem_p02981


def test_problem_p02981_0():
    actual_output = problem_p02981("4 2 9")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02981_1():
    actual_output = problem_p02981("4 2 7")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02981_2():
    actual_output = problem_p02981("4 2 9")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02981_3():
    actual_output = problem_p02981("4 2 8")
    expected_output = "8"
    assert str(actual_output) == expected_output
