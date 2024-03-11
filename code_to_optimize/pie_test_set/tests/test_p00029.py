from code_to_optimize.pie_test_set.p00029 import problem_p00029


def test_problem_p00029_0():
    actual_output = problem_p00029("Thank you for your mail and your lectures")
    expected_output = "your lectures"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00029_1():
    actual_output = problem_p00029("Thank you for your mail and your lectures")
    expected_output = "your lectures"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
