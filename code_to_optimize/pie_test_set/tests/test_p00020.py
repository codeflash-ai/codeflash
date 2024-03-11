from code_to_optimize.pie_test_set.p00020 import problem_p00020


def test_problem_p00020_0():
    actual_output = problem_p00020("this is a pen.")
    expected_output = "THIS IS A PEN."
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00020_1():
    actual_output = problem_p00020("this is a pen.")
    expected_output = "THIS IS A PEN."
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
