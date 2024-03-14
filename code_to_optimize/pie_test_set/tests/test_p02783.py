from code_to_optimize.pie_test_set.p02783 import problem_p02783


def test_problem_p02783_0():
    actual_output = problem_p02783("10 4")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02783_1():
    actual_output = problem_p02783("10 4")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02783_2():
    actual_output = problem_p02783("1 10000")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02783_3():
    actual_output = problem_p02783("10000 1")
    expected_output = "10000"
    assert str(actual_output) == expected_output
