from code_to_optimize.pie_test_set.p04033 import problem_p04033


def test_problem_p04033_0():
    actual_output = problem_p04033("1 3")
    expected_output = "Positive"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
