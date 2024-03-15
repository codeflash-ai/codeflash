from code_to_optimize.pie_test_set.p02400 import problem_p02400


def test_problem_p02400_0():
    actual_output = problem_p02400("2")
    expected_output = "12.566371 12.566371"
    assert str(actual_output) == expected_output


def test_problem_p02400_1():
    actual_output = problem_p02400("2")
    expected_output = "12.566371 12.566371"
    assert str(actual_output) == expected_output


def test_problem_p02400_2():
    actual_output = problem_p02400("3")
    expected_output = "28.274334 18.849556"
    assert str(actual_output) == expected_output
