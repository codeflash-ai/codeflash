from pie_test_set.p02853 import problem_p02853


def test_problem_p02853_0():
    actual_output = problem_p02853("1 1")
    expected_output = "1000000"
    assert str(actual_output) == expected_output


def test_problem_p02853_1():
    actual_output = problem_p02853("3 101")
    expected_output = "100000"
    assert str(actual_output) == expected_output


def test_problem_p02853_2():
    actual_output = problem_p02853("1 1")
    expected_output = "1000000"
    assert str(actual_output) == expected_output


def test_problem_p02853_3():
    actual_output = problem_p02853("4 4")
    expected_output = "0"
    assert str(actual_output) == expected_output
