from code_to_optimize.pie_test_set.p02782 import problem_p02782


def test_problem_p02782_0():
    actual_output = problem_p02782("1 1 2 2")
    expected_output = "14"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02782_1():
    actual_output = problem_p02782("314 159 2653 589")
    expected_output = "602215194"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02782_2():
    actual_output = problem_p02782("1 1 2 2")
    expected_output = "14"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
