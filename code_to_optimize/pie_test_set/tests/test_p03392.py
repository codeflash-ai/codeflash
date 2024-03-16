from code_to_optimize.pie_test_set.p03392 import problem_p03392


def test_problem_p03392_0():
    actual_output = problem_p03392("abc")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03392_1():
    actual_output = problem_p03392("babacabac")
    expected_output = "6310"
    assert str(actual_output) == expected_output


def test_problem_p03392_2():
    actual_output = problem_p03392("abbac")
    expected_output = "65"
    assert str(actual_output) == expected_output


def test_problem_p03392_3():
    actual_output = problem_p03392("abc")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p03392_4():
    actual_output = problem_p03392("ababacbcacbacacbcbbcbbacbaccacbacbacba")
    expected_output = "148010497"
    assert str(actual_output) == expected_output
