from pie_test_set.p03573 import problem_p03573


def test_problem_p03573_0():
    actual_output = problem_p03573("5 7 5")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03573_1():
    actual_output = problem_p03573("5 7 5")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03573_2():
    actual_output = problem_p03573("-100 100 100")
    expected_output = "-100"
    assert str(actual_output) == expected_output


def test_problem_p03573_3():
    actual_output = problem_p03573("1 1 7")
    expected_output = "7"
    assert str(actual_output) == expected_output
