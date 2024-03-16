from code_to_optimize.pie_test_set.p03427 import problem_p03427


def test_problem_p03427_0():
    actual_output = problem_p03427("100")
    expected_output = "18"
    assert str(actual_output) == expected_output


def test_problem_p03427_1():
    actual_output = problem_p03427("3141592653589793")
    expected_output = "137"
    assert str(actual_output) == expected_output


def test_problem_p03427_2():
    actual_output = problem_p03427("100")
    expected_output = "18"
    assert str(actual_output) == expected_output


def test_problem_p03427_3():
    actual_output = problem_p03427("9995")
    expected_output = "35"
    assert str(actual_output) == expected_output
