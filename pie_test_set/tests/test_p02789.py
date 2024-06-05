from pie_test_set.p02789 import problem_p02789


def test_problem_p02789_0():
    actual_output = problem_p02789("3 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02789_1():
    actual_output = problem_p02789("3 3")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02789_2():
    actual_output = problem_p02789("1 1")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02789_3():
    actual_output = problem_p02789("3 2")
    expected_output = "No"
    assert str(actual_output) == expected_output
