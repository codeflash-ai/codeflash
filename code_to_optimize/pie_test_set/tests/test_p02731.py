from code_to_optimize.pie_test_set.p02731 import problem_p02731


def test_problem_p02731_0():
    actual_output = problem_p02731("3")
    expected_output = "1.000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02731_1():
    actual_output = problem_p02731("999")
    expected_output = "36926037.000000000000"
    assert str(actual_output) == expected_output


def test_problem_p02731_2():
    actual_output = problem_p02731("3")
    expected_output = "1.000000000000"
    assert str(actual_output) == expected_output
