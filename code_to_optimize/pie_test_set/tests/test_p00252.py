from code_to_optimize.pie_test_set.p00252 import problem_p00252


def test_problem_p00252_0():
    actual_output = problem_p00252("0 0 1")
    expected_output = "Open"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00252_1():
    actual_output = problem_p00252("0 0 1")
    expected_output = "Open"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
