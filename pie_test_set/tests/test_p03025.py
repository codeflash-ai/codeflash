from pie_test_set.p03025 import problem_p03025


def test_problem_p03025_0():
    actual_output = problem_p03025("1 25 25 50")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03025_1():
    actual_output = problem_p03025("4 50 50 0")
    expected_output = "312500008"
    assert str(actual_output) == expected_output


def test_problem_p03025_2():
    actual_output = problem_p03025("1 25 25 50")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03025_3():
    actual_output = problem_p03025("1 100 0 0")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03025_4():
    actual_output = problem_p03025("100000 31 41 28")
    expected_output = "104136146"
    assert str(actual_output) == expected_output
