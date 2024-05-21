from pie_test_set.p02640 import problem_p02640


def test_problem_p02640_0():
    actual_output = problem_p02640("3 8")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02640_1():
    actual_output = problem_p02640("1 2")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02640_2():
    actual_output = problem_p02640("3 8")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02640_3():
    actual_output = problem_p02640("2 100")
    expected_output = "No"
    assert str(actual_output) == expected_output
