from pie_test_set.p02775 import problem_p02775


def test_problem_p02775_0():
    actual_output = problem_p02775("36")
    expected_output = "8"
    assert str(actual_output) == expected_output


def test_problem_p02775_1():
    actual_output = problem_p02775("91")
    expected_output = "3"
    assert str(actual_output) == expected_output


def test_problem_p02775_2():
    actual_output = problem_p02775(
        "314159265358979323846264338327950288419716939937551058209749445923078164062862089986280348253421170"
    )
    expected_output = "243"
    assert str(actual_output) == expected_output


def test_problem_p02775_3():
    actual_output = problem_p02775("36")
    expected_output = "8"
    assert str(actual_output) == expected_output
