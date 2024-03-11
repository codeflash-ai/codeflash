from code_to_optimize.pie_test_set.p01486 import problem_p01486


def test_problem_p01486_0():
    actual_output = problem_p01486("mmemewwemeww")
    expected_output = "Cat"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p01486_1():
    actual_output = problem_p01486("mmemewwemeww")
    expected_output = "Cat"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
