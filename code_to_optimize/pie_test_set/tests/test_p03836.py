from code_to_optimize.pie_test_set.p03836 import problem_p03836


def test_problem_p03836_0():
    actual_output = problem_p03836("0 0 1 2")
    expected_output = "UURDDLLUUURRDRDDDLLU"
    assert str(actual_output) == expected_output


def test_problem_p03836_1():
    actual_output = problem_p03836("0 0 1 2")
    expected_output = "UURDDLLUUURRDRDDDLLU"
    assert str(actual_output) == expected_output


def test_problem_p03836_2():
    actual_output = problem_p03836("-2 -2 1 1")
    expected_output = "UURRURRDDDLLDLLULUUURRURRDDDLLDL"
    assert str(actual_output) == expected_output
