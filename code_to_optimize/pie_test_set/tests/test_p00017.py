from code_to_optimize.pie_test_set.p00017 import problem_p00017


def test_problem_p00017_0():
    actual_output = problem_p00017("xlmw mw xli tmgxyvi xlex m xsso mr xli xvmt.")
    expected_output = "this is the picture that i took in the trip."
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        print(actual_output)
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p00017_1():
    actual_output = problem_p00017("xlmw mw xli tmgxyvi xlex m xsso mr xli xvmt.")
    expected_output = "this is the picture that i took in the trip."
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
