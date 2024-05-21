from pie_test_set.p02392 import problem_p02392


def test_problem_p02392_0():
    actual_output = problem_p02392("1 3 8")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02392_1():
    actual_output = problem_p02392("1 3 8")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02392_2():
    actual_output = problem_p02392("3 8 1")
    expected_output = "No"
    assert str(actual_output) == expected_output
