from code_to_optimize.pie_test_set.p02596 import problem_p02596


def test_problem_p02596_0():
    actual_output = problem_p02596("101")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02596_1():
    actual_output = problem_p02596("2")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p02596_2():
    actual_output = problem_p02596("999983")
    expected_output = "999982"
    assert str(actual_output) == expected_output


def test_problem_p02596_3():
    actual_output = problem_p02596("101")
    expected_output = "4"
    assert str(actual_output) == expected_output
