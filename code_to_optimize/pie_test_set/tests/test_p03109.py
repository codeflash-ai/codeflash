from code_to_optimize.pie_test_set.p03109 import problem_p03109


def test_problem_p03109_0():
    actual_output = problem_p03109("2019/04/30")
    expected_output = "Heisei"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03109_1():
    actual_output = problem_p03109("2019/04/30")
    expected_output = "Heisei"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03109_2():
    actual_output = problem_p03109("2019/11/01")
    expected_output = "TBD"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
