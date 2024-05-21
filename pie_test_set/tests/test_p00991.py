from pie_test_set.p00991 import problem_p00991


def test_problem_p00991_0():
    actual_output = problem_p00991("4 4 0 0 3 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p00991_1():
    actual_output = problem_p00991("500 500 0 0 200 200")
    expected_output = "34807775"
    assert str(actual_output) == expected_output


def test_problem_p00991_2():
    actual_output = problem_p00991("4 4 0 0 1 1")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p00991_3():
    actual_output = problem_p00991("4 4 0 0 3 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p00991_4():
    actual_output = problem_p00991("2 3 0 0 1 2")
    expected_output = "4"
    assert str(actual_output) == expected_output
