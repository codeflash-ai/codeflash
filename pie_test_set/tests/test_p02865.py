from pie_test_set.p02865 import problem_p02865


def test_problem_p02865_0():
    actual_output = problem_p02865("4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02865_1():
    actual_output = problem_p02865("4")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02865_2():
    actual_output = problem_p02865("999999")
    expected_output = "499999"
    assert str(actual_output) == expected_output
