from pie_test_set.p02938 import problem_p02938


def test_problem_p02938_0():
    actual_output = problem_p02938("2 3")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02938_1():
    actual_output = problem_p02938("2 3")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02938_2():
    actual_output = problem_p02938("10 100")
    expected_output = "604"
    assert str(actual_output) == expected_output


def test_problem_p02938_3():
    actual_output = problem_p02938("1 1000000000000000000")
    expected_output = "68038601"
    assert str(actual_output) == expected_output
