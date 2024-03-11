from code_to_optimize.pie_test_set.p01772 import problem_p01772


def test_problem_p01772_0():
    actual_output = problem_p01772("AIZUNYANPEROPERO")
    expected_output = "AZ"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p01772_1():
    actual_output = problem_p01772("AIZUNYANPEROPERO")
    expected_output = "AZ"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
