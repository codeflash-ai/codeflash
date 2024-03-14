from code_to_optimize.pie_test_set.p02723 import problem_p02723


def test_problem_p02723_0():
    actual_output = problem_p02723("sippuu")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02723_1():
    actual_output = problem_p02723("coffee")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02723_2():
    actual_output = problem_p02723("sippuu")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02723_3():
    actual_output = problem_p02723("iphone")
    expected_output = "No"
    assert str(actual_output) == expected_output
