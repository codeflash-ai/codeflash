from code_to_optimize.pie_test_set.p02624 import problem_p02624


def test_problem_p02624_0():
    actual_output = problem_p02624("4")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02624_1():
    actual_output = problem_p02624("4")
    expected_output = "23"
    assert str(actual_output) == expected_output


def test_problem_p02624_2():
    actual_output = problem_p02624("100")
    expected_output = "26879"
    assert str(actual_output) == expected_output


def test_problem_p02624_3():
    actual_output = problem_p02624("10000000")
    expected_output = "838627288460105"
    assert str(actual_output) == expected_output
