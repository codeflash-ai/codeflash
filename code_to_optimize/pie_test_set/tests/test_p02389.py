from code_to_optimize.pie_test_set.p02389 import problem_p02389


def test_problem_p02389_0():
    actual_output = problem_p02389("3 5")
    expected_output = "15 16"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02389_1():
    actual_output = problem_p02389("3 5")
    expected_output = "15 16"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
