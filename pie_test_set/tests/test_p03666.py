from pie_test_set.p03666 import problem_p03666


def test_problem_p03666_0():
    actual_output = problem_p03666("5 1 5 2 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03666_1():
    actual_output = problem_p03666("48792 105960835 681218449 90629745 90632170")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03666_2():
    actual_output = problem_p03666("491995 412925347 825318103 59999126 59999339")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03666_3():
    actual_output = problem_p03666("5 1 5 2 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03666_4():
    actual_output = problem_p03666("4 7 6 4 5")
    expected_output = "NO"
    assert str(actual_output) == expected_output
