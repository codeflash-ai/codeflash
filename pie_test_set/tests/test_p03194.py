from pie_test_set.p03194 import problem_p03194


def test_problem_p03194_0():
    actual_output = problem_p03194("3 24")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03194_1():
    actual_output = problem_p03194("5 1")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p03194_2():
    actual_output = problem_p03194("4 972439611840")
    expected_output = "206"
    assert str(actual_output) == expected_output


def test_problem_p03194_3():
    actual_output = problem_p03194("3 24")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03194_4():
    actual_output = problem_p03194("1 111")
    expected_output = "111"
    assert str(actual_output) == expected_output
