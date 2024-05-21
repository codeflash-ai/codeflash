from pie_test_set.p03939 import problem_p03939


def test_problem_p03939_0():
    actual_output = problem_p03939("1 3 3")
    expected_output = "4.500000000000000"
    assert str(actual_output) == expected_output


def test_problem_p03939_1():
    actual_output = problem_p03939("1 3 3")
    expected_output = "4.500000000000000"
    assert str(actual_output) == expected_output


def test_problem_p03939_2():
    actual_output = problem_p03939("1000 100 100")
    expected_output = "649620280.957660079002380"
    assert str(actual_output) == expected_output


def test_problem_p03939_3():
    actual_output = problem_p03939("2 1 0")
    expected_output = "2.500000000000000"
    assert str(actual_output) == expected_output
