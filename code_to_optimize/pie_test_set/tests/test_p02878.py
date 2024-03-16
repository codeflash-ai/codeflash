from code_to_optimize.pie_test_set.p02878 import problem_p02878


def test_problem_p02878_0():
    actual_output = problem_p02878("5 1 3")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02878_1():
    actual_output = problem_p02878("5 1 3")
    expected_output = "4"
    assert str(actual_output) == expected_output


def test_problem_p02878_2():
    actual_output = problem_p02878("1000000 100000 200000")
    expected_output = "758840509"
    assert str(actual_output) == expected_output


def test_problem_p02878_3():
    actual_output = problem_p02878("10 4 6")
    expected_output = "197"
    assert str(actual_output) == expected_output


def test_problem_p02878_4():
    actual_output = problem_p02878("10 0 0")
    expected_output = "1"
    assert str(actual_output) == expected_output
