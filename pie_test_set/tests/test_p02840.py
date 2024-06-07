from pie_test_set.p02840 import problem_p02840


def test_problem_p02840_0():
    actual_output = problem_p02840("3 4 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02840_1():
    actual_output = problem_p02840("2 3 -3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02840_2():
    actual_output = problem_p02840("3 4 2")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02840_3():
    actual_output = problem_p02840("100 14 20")
    expected_output = "49805"
    assert str(actual_output) == expected_output
