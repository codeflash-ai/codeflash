from code_to_optimize.pie_test_set.p02550 import problem_p02550


def test_problem_p02550_0():
    actual_output = problem_p02550("6 2 1001")
    expected_output = "1369"
    assert str(actual_output) == expected_output


def test_problem_p02550_1():
    actual_output = problem_p02550("10000000000 10 99959")
    expected_output = "492443256176507"
    assert str(actual_output) == expected_output


def test_problem_p02550_2():
    actual_output = problem_p02550("6 2 1001")
    expected_output = "1369"
    assert str(actual_output) == expected_output


def test_problem_p02550_3():
    actual_output = problem_p02550("1000 2 16")
    expected_output = "6"
    assert str(actual_output) == expected_output
