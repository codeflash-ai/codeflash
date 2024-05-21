from pie_test_set.p02880 import problem_p02880


def test_problem_p02880_0():
    actual_output = problem_p02880("10")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02880_1():
    actual_output = problem_p02880("10")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02880_2():
    actual_output = problem_p02880("81")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02880_3():
    actual_output = problem_p02880("50")
    expected_output = "No"
    assert str(actual_output) == expected_output
