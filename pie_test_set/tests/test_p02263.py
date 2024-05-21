from pie_test_set.p02263 import problem_p02263


def test_problem_p02263_0():
    actual_output = problem_p02263("1 2 +")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02263_1():
    actual_output = problem_p02263("1 2 + 3 4 - *")
    expected_output = "-3"
    assert str(actual_output) == expected_output


def test_problem_p02263_2():
    actual_output = problem_p02263("1 2 +")
    expected_output = "3"
    assert str(actual_output) == expected_output
