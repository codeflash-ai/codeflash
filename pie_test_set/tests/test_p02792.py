from pie_test_set.p02792 import problem_p02792


def test_problem_p02792_0():
    actual_output = problem_p02792("25")
    expected_output = "17"
    assert str(actual_output) == expected_output


def test_problem_p02792_1():
    actual_output = problem_p02792("2020")
    expected_output = "40812"
    assert str(actual_output) == expected_output


def test_problem_p02792_2():
    actual_output = problem_p02792("200000")
    expected_output = "400000008"
    assert str(actual_output) == expected_output


def test_problem_p02792_3():
    actual_output = problem_p02792("25")
    expected_output = "17"
    assert str(actual_output) == expected_output


def test_problem_p02792_4():
    actual_output = problem_p02792("100")
    expected_output = "108"
    assert str(actual_output) == expected_output


def test_problem_p02792_5():
    actual_output = problem_p02792("1")
    expected_output = "1"
    assert str(actual_output) == expected_output
