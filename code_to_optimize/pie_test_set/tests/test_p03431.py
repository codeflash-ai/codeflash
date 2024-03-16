from code_to_optimize.pie_test_set.p03431 import problem_p03431


def test_problem_p03431_0():
    actual_output = problem_p03431("2 4")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p03431_1():
    actual_output = problem_p03431("8 10")
    expected_output = "46"
    assert str(actual_output) == expected_output


def test_problem_p03431_2():
    actual_output = problem_p03431("3 7")
    expected_output = "57"
    assert str(actual_output) == expected_output


def test_problem_p03431_3():
    actual_output = problem_p03431("123456 234567")
    expected_output = "857617983"
    assert str(actual_output) == expected_output


def test_problem_p03431_4():
    actual_output = problem_p03431("8 3")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p03431_5():
    actual_output = problem_p03431("2 4")
    expected_output = "7"
    assert str(actual_output) == expected_output
