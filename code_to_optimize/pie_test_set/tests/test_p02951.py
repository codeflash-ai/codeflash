from code_to_optimize.pie_test_set.p02951 import problem_p02951


def test_problem_p02951_0():
    actual_output = problem_p02951("6 4 3")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02951_1():
    actual_output = problem_p02951("12 3 7")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02951_2():
    actual_output = problem_p02951("8 3 9")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02951_3():
    actual_output = problem_p02951("6 4 3")
    expected_output = "1"
    assert str(actual_output) == expected_output
