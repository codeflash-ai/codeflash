from code_to_optimize.pie_test_set.p02400 import problem_p02400


def test_problem_p02400_0():
    actual_output = problem_p02400("2")
    expected_output = "12.566371 12.566371"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02400_1():
    actual_output = problem_p02400("2")
    expected_output = "12.566371 12.566371"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02400_2():
    actual_output = problem_p02400("3")
    expected_output = "28.274334 18.849556"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
