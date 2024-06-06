from pie_test_set.p02900 import problem_p02900


def test_problem_p02900_0():
    actual_output = problem_p02900("12 18")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02900_1():
    actual_output = problem_p02900("1 2019")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02900_2():
    actual_output = problem_p02900("12 18")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02900_3():
    actual_output = problem_p02900("420 660")
    expected_output = "4"
    assert str(actual_output) == expected_output
