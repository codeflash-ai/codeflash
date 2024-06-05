from pie_test_set.p02606 import problem_p02606


def test_problem_p02606_0():
    actual_output = problem_p02606("5 10 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02606_1():
    actual_output = problem_p02606("1 100 1")
    expected_output = "100"
    assert str(actual_output) == expected_output


def test_problem_p02606_2():
    actual_output = problem_p02606("6 20 7")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02606_3():
    actual_output = problem_p02606("5 10 2")
    expected_output = "3"
    assert str(actual_output) == expected_output
