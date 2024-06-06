from pie_test_set.p02879 import problem_p02879


def test_problem_p02879_0():
    actual_output = problem_p02879("2 5")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02879_1():
    actual_output = problem_p02879("2 5")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02879_2():
    actual_output = problem_p02879("9 9")
    expected_output = "81"
    assert str(actual_output) == expected_output


def test_problem_p02879_3():
    actual_output = problem_p02879("5 10")
    expected_output = "-1"
    assert str(actual_output) == expected_output
