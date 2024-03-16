from code_to_optimize.pie_test_set.p02987 import problem_p02987


def test_problem_p02987_0():
    actual_output = problem_p02987("ASSA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02987_1():
    actual_output = problem_p02987("FFEE")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02987_2():
    actual_output = problem_p02987("FREE")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02987_3():
    actual_output = problem_p02987("STOP")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02987_4():
    actual_output = problem_p02987("ASSA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
