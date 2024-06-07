from pie_test_set.p02541 import problem_p02541


def test_problem_p02541_0():
    actual_output = problem_p02541("11")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02541_1():
    actual_output = problem_p02541("20200920")
    expected_output = "1100144"
    assert str(actual_output) == expected_output


def test_problem_p02541_2():
    actual_output = problem_p02541("11")
    expected_output = "10"
    assert str(actual_output) == expected_output
