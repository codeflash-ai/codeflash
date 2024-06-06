from pie_test_set.p02715 import problem_p02715


def test_problem_p02715_0():
    actual_output = problem_p02715("3 2")
    expected_output = "9"
    assert str(actual_output) == expected_output


def test_problem_p02715_1():
    actual_output = problem_p02715("100000 100000")
    expected_output = "742202979"
    assert str(actual_output) == expected_output


def test_problem_p02715_2():
    actual_output = problem_p02715("3 200")
    expected_output = "10813692"
    assert str(actual_output) == expected_output


def test_problem_p02715_3():
    actual_output = problem_p02715("3 2")
    expected_output = "9"
    assert str(actual_output) == expected_output
