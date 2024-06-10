from pie_test_set.p03375 import problem_p03375


def test_problem_p03375_0():
    actual_output = problem_p03375("2 1000000007")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03375_1():
    actual_output = problem_p03375("3 1000000009")
    expected_output = "118"
    assert str(actual_output) == expected_output


def test_problem_p03375_2():
    actual_output = problem_p03375("2 1000000007")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03375_3():
    actual_output = problem_p03375("3000 123456791")
    expected_output = "16369789"
    assert str(actual_output) == expected_output


def test_problem_p03375_4():
    actual_output = problem_p03375("50 111111113")
    expected_output = "1456748"
    assert str(actual_output) == expected_output
