from pie_test_set.p02116 import problem_p02116


def test_problem_p02116_0():
    actual_output = problem_p02116("2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02116_1():
    actual_output = problem_p02116("3")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02116_2():
    actual_output = problem_p02116("2")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02116_3():
    actual_output = problem_p02116("111")
    expected_output = "16"
    assert str(actual_output) == expected_output
