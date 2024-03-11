from code_to_optimize.pie_test_set.p00006 import problem_p00006


def test_problem_p00006_0():
    actual_output = problem_p00006("w32nimda")
    expected_output = "admin23w"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00006_1():
    actual_output = problem_p00006("w32nimda")
    expected_output = "admin23w"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
