from code_to_optimize.pie_test_set.p01741 import problem_p01741


def test_problem_p01741_0():
    actual_output = problem_p01741("1.000")
    expected_output = "2.000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p01741_1():
    actual_output = problem_p01741("1.000")
    expected_output = "2.000000000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p01741_2():
    actual_output = problem_p01741("2.345")
    expected_output = "3.316330803765"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
