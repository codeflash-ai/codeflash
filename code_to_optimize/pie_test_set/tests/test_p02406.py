from code_to_optimize.pie_test_set.p02406 import problem_p02406


def test_problem_p02406_0():
    actual_output = problem_p02406("30")
    expected_output = "3 6 9 12 13 15 18 21 23 24 27 30"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02406_1():
    actual_output = problem_p02406("30")
    expected_output = "3 6 9 12 13 15 18 21 23 24 27 30"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
