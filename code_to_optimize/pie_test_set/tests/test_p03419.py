from code_to_optimize.pie_test_set.p03419 import problem_p03419


def test_problem_p03419_0():
    actual_output = problem_p03419("2 2")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03419_1():
    actual_output = problem_p03419("1 7")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p03419_2():
    actual_output = problem_p03419("2 2")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03419_3():
    actual_output = problem_p03419("314 1592")
    expected_output = "496080"
    assert str(actual_output) == expected_output
