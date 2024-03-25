from code_to_optimize.pie_test_set.p02960 import problem_p02960


def test_problem_p02960_0():
    actual_output = problem_p02960("??2??5")
    expected_output = "768"
    assert str(actual_output) == expected_output


def test_problem_p02960_1():
    actual_output = problem_p02960("7?4")
    expected_output = "0"
    assert str(actual_output) == expected_output


def test_problem_p02960_2():
    actual_output = problem_p02960("??2??5")
    expected_output = "768"
    assert str(actual_output) == expected_output


def test_problem_p02960_3():
    actual_output = problem_p02960(
        "?6?42???8??2??06243????9??3???7258??5??7???????774????4?1??17???9?5?70???76???"
    )
    expected_output = "153716888"
    assert str(actual_output) == expected_output


def test_problem_p02960_4():
    actual_output = problem_p02960("?44")
    expected_output = "1"
    assert str(actual_output) == expected_output
