from pie_test_set.p02910 import problem_p02910


def test_problem_p02910_0():
    actual_output = problem_p02910("RUDLUDR")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02910_1():
    actual_output = problem_p02910("RUDLUDR")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02910_2():
    actual_output = problem_p02910("DULL")
    expected_output = "No"
    assert str(actual_output) == expected_output


def test_problem_p02910_3():
    actual_output = problem_p02910("RDULULDURURLRDULRLR")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02910_4():
    actual_output = problem_p02910("UUUUUUUUUUUUUUU")
    expected_output = "Yes"
    assert str(actual_output) == expected_output


def test_problem_p02910_5():
    actual_output = problem_p02910("ULURU")
    expected_output = "No"
    assert str(actual_output) == expected_output
