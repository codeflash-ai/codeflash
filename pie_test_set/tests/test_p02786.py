from pie_test_set.p02786 import problem_p02786


def test_problem_p02786_0():
    actual_output = problem_p02786("2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02786_1():
    actual_output = problem_p02786("4")
    expected_output = "7"
    assert str(actual_output) == expected_output


def test_problem_p02786_2():
    actual_output = problem_p02786("1000000000000")
    expected_output = "1099511627775"
    assert str(actual_output) == expected_output


def test_problem_p02786_3():
    actual_output = problem_p02786("2")
    expected_output = "3"
    assert str(actual_output) == expected_output
