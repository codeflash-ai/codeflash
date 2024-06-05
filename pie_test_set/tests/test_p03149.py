from pie_test_set.p03149 import problem_p03149


def test_problem_p03149_0():
    actual_output = problem_p03149("1 7 9 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03149_1():
    actual_output = problem_p03149("1 9 7 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03149_2():
    actual_output = problem_p03149("1 2 9 1")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03149_3():
    actual_output = problem_p03149("4 9 0 8")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03149_4():
    actual_output = problem_p03149("1 7 9 4")
    expected_output = "YES"
    assert str(actual_output) == expected_output
