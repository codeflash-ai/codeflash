from code_to_optimize.pie_test_set.p02379 import problem_p02379


def test_problem_p02379_0():
    actual_output = problem_p02379("0 0 1 1")
    expected_output = "1.41421356"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02379_1():
    actual_output = problem_p02379("0 0 1 1")
    expected_output = "1.41421356"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
