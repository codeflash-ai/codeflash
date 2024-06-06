from pie_test_set.p02625 import problem_p02625


def test_problem_p02625_0():
    actual_output = problem_p02625("2 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02625_1():
    actual_output = problem_p02625("141421 356237")
    expected_output = "881613484"
    assert str(actual_output) == expected_output


def test_problem_p02625_2():
    actual_output = problem_p02625("2 2")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p02625_3():
    actual_output = problem_p02625("2 3")
    expected_output = "18"
    assert str(actual_output) == expected_output
