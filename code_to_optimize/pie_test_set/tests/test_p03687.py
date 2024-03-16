from code_to_optimize.pie_test_set.p03687 import problem_p03687


def test_problem_p03687_0():
    actual_output = problem_p03687("serval")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03687_1():
    actual_output = problem_p03687("serval")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03687_2():
    actual_output = problem_p03687("whbrjpjyhsrywlqjxdbrbaomnw")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p03687_3():
    actual_output = problem_p03687("jackal")
    expected_output = "2"
    assert str(actual_output) == expected_output


def test_problem_p03687_4():
    actual_output = problem_p03687("zzz")
    expected_output = "0"
    assert str(actual_output) == expected_output
