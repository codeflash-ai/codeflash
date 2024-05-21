from pie_test_set.p02790 import problem_p02790


def test_problem_p02790_0():
    actual_output = problem_p02790("4 3")
    expected_output = "3333"
    assert str(actual_output) == expected_output


def test_problem_p02790_1():
    actual_output = problem_p02790("7 7")
    expected_output = "7777777"
    assert str(actual_output) == expected_output


def test_problem_p02790_2():
    actual_output = problem_p02790("4 3")
    expected_output = "3333"
    assert str(actual_output) == expected_output
