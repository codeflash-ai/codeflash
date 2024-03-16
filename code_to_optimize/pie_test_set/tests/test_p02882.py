from code_to_optimize.pie_test_set.p02882 import problem_p02882


def test_problem_p02882_0():
    actual_output = problem_p02882("2 2 4")
    expected_output = "45.0000000000"
    assert str(actual_output) == expected_output


def test_problem_p02882_1():
    actual_output = problem_p02882("2 2 4")
    expected_output = "45.0000000000"
    assert str(actual_output) == expected_output


def test_problem_p02882_2():
    actual_output = problem_p02882("12 21 10")
    expected_output = "89.7834636934"
    assert str(actual_output) == expected_output


def test_problem_p02882_3():
    actual_output = problem_p02882("3 1 8")
    expected_output = "4.2363947991"
    assert str(actual_output) == expected_output
