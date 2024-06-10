from pie_test_set.p03145 import problem_p03145


def test_problem_p03145_0():
    actual_output = problem_p03145("3 4 5")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03145_1():
    actual_output = problem_p03145("3 4 5")
    expected_output = "6"
    assert str(actual_output) == expected_output


def test_problem_p03145_2():
    actual_output = problem_p03145("45 28 53")
    expected_output = "630"
    assert str(actual_output) == expected_output


def test_problem_p03145_3():
    actual_output = problem_p03145("5 12 13")
    expected_output = "30"
    assert str(actual_output) == expected_output
