from code_to_optimize.pie_test_set.p02694 import problem_p02694


def test_problem_p02694_0():
    actual_output = problem_p02694("103")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02694_1():
    actual_output = problem_p02694("1333333333")
    expected_output = "1706"
    assert str(actual_output) == expected_output


def test_problem_p02694_2():
    actual_output = problem_p02694("103")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02694_3():
    actual_output = problem_p02694("1000000000000000000")
    expected_output = "3760"
    assert str(actual_output) == expected_output
