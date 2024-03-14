from code_to_optimize.pie_test_set.p02730 import problem_p02730


def test_problem_p02730_0():
    actual_output = problem_p02730("akasaka")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02730_1():
    actual_output = problem_p02730("akasaka")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02730_2():
    actual_output = problem_p02730("atcoder")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02730_3():
    actual_output = problem_p02730("level")
    expected_output = "No"
    assert str(actual_output) == expected_output
