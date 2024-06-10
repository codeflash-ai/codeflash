from pie_test_set.p02555 import problem_p02555


def test_problem_p02555_0():
    actual_output = problem_p02555("7")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02555_1():
    actual_output = problem_p02555("7")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02555_2():
    actual_output = problem_p02555("1729")
    expected_output = "294867501"
    assert str(actual_output) == expected_output


def test_problem_p02555_3():
    actual_output = problem_p02555("2")
    expected_output = "0"
    assert str(actual_output) == expected_output
