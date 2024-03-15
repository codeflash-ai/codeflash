from code_to_optimize.pie_test_set.p02965 import problem_p02965


def test_problem_p02965_0():
    actual_output = problem_p02965("2 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02965_1():
    actual_output = problem_p02965("3 2")
    expected_output = "19"
    assert str(actual_output) == expected_output


def test_problem_p02965_2():
    actual_output = problem_p02965("100000 50000")
    expected_output = "3463133"
    assert str(actual_output) == expected_output


def test_problem_p02965_3():
    actual_output = problem_p02965("10 10")
    expected_output = "211428932"
    assert str(actual_output) == expected_output


def test_problem_p02965_4():
    actual_output = problem_p02965("2 2")
    expected_output = "3"
    assert str(actual_output) == expected_output
