from pie_test_set.p03050 import problem_p03050


def test_problem_p03050_0():
    actual_output = problem_p03050("8")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03050_1():
    actual_output = problem_p03050("8")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p03050_2():
    actual_output = problem_p03050("1000000000000")
    expected_output = "2499686339916"
    assert str(actual_output) == expected_output
