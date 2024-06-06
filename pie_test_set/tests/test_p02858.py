from pie_test_set.p02858 import problem_p02858


def test_problem_p02858_0():
    actual_output = problem_p02858("2 2 1")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02858_1():
    actual_output = problem_p02858("869 120 1001")
    expected_output = "672919729"
    assert str(actual_output) == expected_output


def test_problem_p02858_2():
    actual_output = problem_p02858("2 2 1")
    expected_output = "9"
    assert str(actual_output) == expected_output
