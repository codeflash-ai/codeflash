from pie_test_set.p02753 import problem_p02753


def test_problem_p02753_0():
    actual_output = problem_p02753("ABA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02753_1():
    actual_output = problem_p02753("ABA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02753_2():
    actual_output = problem_p02753("BBA")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02753_3():
    actual_output = problem_p02753("BBB")
    expected_output = "No"
    assert str(actual_output) == expected_output
