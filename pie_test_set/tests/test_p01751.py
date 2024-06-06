from pie_test_set.p01751 import problem_p01751


def test_problem_p01751_0():
    actual_output = problem_p01751("10 10 5")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p01751_1():
    actual_output = problem_p01751("20 20 20")
    expected_output = "20"
    assert str(actual_output) == expected_output


def test_problem_p01751_2():
    actual_output = problem_p01751("50 40 51")
    expected_output = "111"
    assert str(actual_output) == expected_output


def test_problem_p01751_3():
    actual_output = problem_p01751("10 10 5")
    expected_output = "5"
    assert str(actual_output) == expected_output


def test_problem_p01751_4():
    actual_output = problem_p01751("30 30 40")
    expected_output = "-1"
    assert str(actual_output) == expected_output
