from pie_test_set.p03636 import problem_p03636


def test_problem_p03636_0():
    actual_output = problem_p03636("internationalization")
    expected_output = "i18n"
    assert str(actual_output) == expected_output


def test_problem_p03636_1():
    actual_output = problem_p03636("smiles")
    expected_output = "s4s"
    assert str(actual_output) == expected_output


def test_problem_p03636_2():
    actual_output = problem_p03636("internationalization")
    expected_output = "i18n"
    assert str(actual_output) == expected_output


def test_problem_p03636_3():
    actual_output = problem_p03636("xyz")
    expected_output = "x1z"
    assert str(actual_output) == expected_output
