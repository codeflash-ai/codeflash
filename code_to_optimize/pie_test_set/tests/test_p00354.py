from code_to_optimize.pie_test_set.p00354 import problem_p00354


def test_problem_p00354_0():
    actual_output = problem_p00354("1")
    expected_output = "fri"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
