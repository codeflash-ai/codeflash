from code_to_optimize.pie_test_set.p02708 import problem_p02708


def test_problem_p02708_0():
    actual_output = problem_p02708("3 2")
    expected_output = "10"
    assert str(actual_output) == expected_output


def test_problem_p02708_1():
    actual_output = problem_p02708("141421 35623")
    expected_output = "220280457"
    assert str(actual_output) == expected_output


def test_problem_p02708_2():
    actual_output = problem_p02708("200000 200001")
    expected_output = "1"
    assert str(actual_output) == expected_output


def test_problem_p02708_3():
    actual_output = problem_p02708("3 2")
    expected_output = "10"
    assert str(actual_output) == expected_output
