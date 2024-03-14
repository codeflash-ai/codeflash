from code_to_optimize.pie_test_set.p02699 import problem_p02699


def test_problem_p02699_0():
    actual_output = problem_p02699("4 5")
    expected_output = "unsafe"
    assert str(actual_output) == expected_output


def test_problem_p02699_1():
    actual_output = problem_p02699("4 5")
    expected_output = "unsafe"
    assert str(actual_output) == expected_output


def test_problem_p02699_2():
    actual_output = problem_p02699("100 2")
    expected_output = "safe"
    assert str(actual_output) == expected_output


def test_problem_p02699_3():
    actual_output = problem_p02699("10 10")
    expected_output = "unsafe"
    assert str(actual_output) == expected_output
