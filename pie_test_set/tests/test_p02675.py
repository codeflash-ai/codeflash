from pie_test_set.p02675 import problem_p02675


def test_problem_p02675_0():
    actual_output = problem_p02675("16")
    expected_output = "pon"
    assert str(actual_output) == expected_output


def test_problem_p02675_1():
    actual_output = problem_p02675("16")
    expected_output = "pon"
    assert str(actual_output) == expected_output


def test_problem_p02675_2():
    actual_output = problem_p02675("2")
    expected_output = "hon"
    assert str(actual_output) == expected_output


def test_problem_p02675_3():
    actual_output = problem_p02675("183")
    expected_output = "bon"
    assert str(actual_output) == expected_output
