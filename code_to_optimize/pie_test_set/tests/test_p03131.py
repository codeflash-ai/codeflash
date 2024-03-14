from code_to_optimize.pie_test_set.p03131 import problem_p03131


def test_problem_p03131_0():
    actual_output = problem_p03131("4 2 6")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03131_1():
    actual_output = problem_p03131("4 2 6")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03131_2():
    actual_output = problem_p03131("7 3 4")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03131_3():
    actual_output = problem_p03131("314159265 35897932 384626433")
    expected_output = "48518828981938099"
    assert str(actual_output) == expected_output
