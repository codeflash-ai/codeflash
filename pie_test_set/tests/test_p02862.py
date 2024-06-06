from pie_test_set.p02862 import problem_p02862


def test_problem_p02862_0():
    actual_output = problem_p02862("3 3")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02862_1():
    actual_output = problem_p02862("999999 999999")
    expected_output = "151840682"
    assert str(actual_output) == expected_output


def test_problem_p02862_2():
    actual_output = problem_p02862("2 2")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02862_3():
    actual_output = problem_p02862("3 3")
    expected_output = "2"
    assert str(actual_output) == expected_output
