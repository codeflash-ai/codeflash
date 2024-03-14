from code_to_optimize.pie_test_set.p02685 import problem_p02685


def test_problem_p02685_0():
    actual_output = problem_p02685("3 2 1")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p02685_1():
    actual_output = problem_p02685("100 100 0")
    expected_output = "73074801"
    assert str(actual_output) == expected_output


def test_problem_p02685_2():
    actual_output = problem_p02685("3 2 1")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p02685_3():
    actual_output = problem_p02685("60522 114575 7559")
    expected_output = "479519525"
    assert str(actual_output) == expected_output
