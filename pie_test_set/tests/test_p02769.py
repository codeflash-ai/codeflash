from pie_test_set.p02769 import problem_p02769


def test_problem_p02769_0():
    actual_output = problem_p02769("3 2")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02769_1():
    actual_output = problem_p02769("3 2")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02769_2():
    actual_output = problem_p02769("200000 1000000000")
    expected_output = "607923868"
    assert str(actual_output) == expected_output


def test_problem_p02769_3():
    actual_output = problem_p02769("15 6")
    expected_output = "22583772"
    assert str(actual_output) == expected_output
