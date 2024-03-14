from code_to_optimize.pie_test_set.p02582 import problem_p02582


def test_problem_p02582_0():
    actual_output = problem_p02582("RRS")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02582_1():
    actual_output = problem_p02582("RSR")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02582_2():
    actual_output = problem_p02582("RRS")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02582_3():
    actual_output = problem_p02582("SSS")
    expected_output = "0"
    assert str(actual_output) == expected_output
