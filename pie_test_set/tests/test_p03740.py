from pie_test_set.p03740 import problem_p03740


def test_problem_p03740_0():
    actual_output = problem_p03740("2 1")
    expected_output = "Brown"
    assert str(actual_output) == expected_output


def test_problem_p03740_1():
    actual_output = problem_p03740("0 0")
    expected_output = "Brown"
    assert str(actual_output) == expected_output


def test_problem_p03740_2():
    actual_output = problem_p03740("5 0")
    expected_output = "Alice"
    assert str(actual_output) == expected_output


def test_problem_p03740_3():
    actual_output = problem_p03740("2 1")
    expected_output = "Brown"
    assert str(actual_output) == expected_output


def test_problem_p03740_4():
    actual_output = problem_p03740("4 8")
    expected_output = "Alice"
    assert str(actual_output) == expected_output
