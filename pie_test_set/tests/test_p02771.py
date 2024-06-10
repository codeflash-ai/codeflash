from pie_test_set.p02771 import problem_p02771


def test_problem_p02771_0():
    actual_output = problem_p02771("5 7 5")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02771_1():
    actual_output = problem_p02771("3 3 4")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02771_2():
    actual_output = problem_p02771("4 4 4")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02771_3():
    actual_output = problem_p02771("5 7 5")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02771_4():
    actual_output = problem_p02771("4 9 6")
    expected_output = "No"
    assert str(actual_output) == expected_output
