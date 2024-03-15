from code_to_optimize.pie_test_set.p03698 import problem_p03698


def test_problem_p03698_0():
    actual_output = problem_p03698("uncopyrightable")
    expected_output = "yes"
    assert str(actual_output) == expected_output


def test_problem_p03698_1():
    actual_output = problem_p03698("uncopyrightable")
    expected_output = "yes"
    assert str(actual_output) == expected_output


def test_problem_p03698_2():
    actual_output = problem_p03698("different")
    expected_output = "no"
    assert str(actual_output) == expected_output


def test_problem_p03698_3():
    actual_output = problem_p03698("no")
    expected_output = "yes"
    assert str(actual_output) == expected_output
