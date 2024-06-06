from pie_test_set.p02952 import problem_p02952


def test_problem_p02952_0():
    actual_output = problem_p02952("11")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02952_1():
    actual_output = problem_p02952("11")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02952_2():
    actual_output = problem_p02952("100000")
    expected_output = "90909"
    assert str(actual_output) == expected_output


def test_problem_p02952_3():
    actual_output = problem_p02952("136")
    expected_output = "46"
    assert str(actual_output) == expected_output
