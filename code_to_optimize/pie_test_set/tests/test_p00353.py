from code_to_optimize.pie_test_set.p00353 import problem_p00353


def test_problem_p00353_0():
    actual_output = problem_p00353("1000 3000 3000")
    expected_output = "2000"
    assert str(actual_output) == expected_output


def test_problem_p00353_1():
    actual_output = problem_p00353("5000 3000 4500")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p00353_2():
    actual_output = problem_p00353("1000 3000 3000")
    expected_output = "2000"
    assert str(actual_output) == expected_output


def test_problem_p00353_3():
    actual_output = problem_p00353("500 1000 2000")
    expected_output = "NA"
    assert str(actual_output) == expected_output
