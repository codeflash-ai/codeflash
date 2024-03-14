from code_to_optimize.pie_test_set.p02634 import problem_p02634


def test_problem_p02634_0():
    actual_output = problem_p02634("1 1 2 2")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02634_1():
    actual_output = problem_p02634("2 1 3 4")
    expected_output = "65"
    assert str(actual_output) == expected_output


def test_problem_p02634_2():
    actual_output = problem_p02634("31 41 59 265")
    expected_output = "387222020"
    assert str(actual_output) == expected_output


def test_problem_p02634_3():
    actual_output = problem_p02634("1 1 2 2")
    expected_output = "3"
    assert str(actual_output) == expected_output
