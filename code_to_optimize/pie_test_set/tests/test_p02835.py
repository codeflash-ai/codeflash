from code_to_optimize.pie_test_set.p02835 import problem_p02835


def test_problem_p02835_0():
    actual_output = problem_p02835("5 7 9")
    expected_output = "win"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02835_1():
    actual_output = problem_p02835("5 7 9")
    expected_output = "win"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02835_2():
    actual_output = problem_p02835("13 7 2")
    expected_output = "bust"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
