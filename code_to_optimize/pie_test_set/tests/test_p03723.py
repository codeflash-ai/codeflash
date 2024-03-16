from code_to_optimize.pie_test_set.p03723 import problem_p03723


def test_problem_p03723_0():
    actual_output = problem_p03723("4 12 20")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03723_1():
    actual_output = problem_p03723("14 14 14")
    expected_output = "-1"
    assert str(actual_output) == expected_output


def test_problem_p03723_2():
    actual_output = problem_p03723("454 414 444")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03723_3():
    actual_output = problem_p03723("4 12 20")
    expected_output = "3"
    assert str(actual_output) == expected_output
