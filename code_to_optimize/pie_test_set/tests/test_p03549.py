from code_to_optimize.pie_test_set.p03549 import problem_p03549


def test_problem_p03549_0():
    actual_output = problem_p03549("1 1")
    expected_output = "3800"
    assert str(actual_output) == expected_output


def test_problem_p03549_1():
    actual_output = problem_p03549("100 5")
    expected_output = "608000"
    assert str(actual_output) == expected_output


def test_problem_p03549_2():
    actual_output = problem_p03549("10 2")
    expected_output = "18400"
    assert str(actual_output) == expected_output


def test_problem_p03549_3():
    actual_output = problem_p03549("1 1")
    expected_output = "3800"
    assert str(actual_output) == expected_output
