from pie_test_set.p02747 import problem_p02747


def test_problem_p02747_0():
    actual_output = problem_p02747("hihi")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02747_1():
    actual_output = problem_p02747("ha")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02747_2():
    actual_output = problem_p02747("hi")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02747_3():
    actual_output = problem_p02747("hihi")
    expected_output = "Yes"
    assert str(actual_output) == expected_output
