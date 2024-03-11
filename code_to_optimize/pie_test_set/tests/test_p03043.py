from code_to_optimize.pie_test_set.p03043 import problem_p03043


def test_problem_p03043_0():
    actual_output = problem_p03043("3 10")
    expected_output = "0.145833333333"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03043_1():
    actual_output = problem_p03043("3 10")
    expected_output = "0.145833333333"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03043_2():
    actual_output = problem_p03043("100000 5")
    expected_output = "0.999973749998"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
