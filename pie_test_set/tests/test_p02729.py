from pie_test_set.p02729 import problem_p02729


def test_problem_p02729_0():
    actual_output = problem_p02729("2 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02729_1():
    actual_output = problem_p02729("13 3")
    expected_output = "81"
    assert str(actual_output) == expected_output


def test_problem_p02729_2():
    actual_output = problem_p02729("1 1")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02729_3():
    actual_output = problem_p02729("4 3")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02729_4():
    actual_output = problem_p02729("2 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02729_5():
    actual_output = problem_p02729("0 3")
    expected_output = "3"
    assert str(actual_output) == expected_output
