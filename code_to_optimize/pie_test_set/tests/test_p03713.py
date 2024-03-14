from code_to_optimize.pie_test_set.p03713 import problem_p03713


def test_problem_p03713_0():
    actual_output = problem_p03713("3 5")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03713_1():
    actual_output = problem_p03713("3 5")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03713_2():
    actual_output = problem_p03713("100000 100000")
    expected_output = "50000"
    assert str(actual_output) == expected_output


def test_problem_p03713_3():
    actual_output = problem_p03713("100000 2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03713_4():
    actual_output = problem_p03713("5 5")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p03713_5():
    actual_output = problem_p03713("4 5")
    expected_output = "2"
    assert str(actual_output) == expected_output
