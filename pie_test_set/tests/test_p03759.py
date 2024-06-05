from pie_test_set.p03759 import problem_p03759


def test_problem_p03759_0():
    actual_output = problem_p03759("2 4 6")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03759_1():
    actual_output = problem_p03759("2 5 6")
    expected_output = "NO"
    assert str(actual_output) == expected_output


def test_problem_p03759_2():
    actual_output = problem_p03759("3 2 1")
    expected_output = "YES"
    assert str(actual_output) == expected_output


def test_problem_p03759_3():
    actual_output = problem_p03759("2 4 6")
    expected_output = "YES"
    assert str(actual_output) == expected_output
