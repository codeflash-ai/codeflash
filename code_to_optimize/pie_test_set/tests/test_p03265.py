from code_to_optimize.pie_test_set.p03265 import problem_p03265


def test_problem_p03265_0():
    actual_output = problem_p03265("0 0 0 1")
    expected_output = "-1 1 -1 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03265_1():
    actual_output = problem_p03265("2 3 6 6")
    expected_output = "3 10 -1 7"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03265_2():
    actual_output = problem_p03265("0 0 0 1")
    expected_output = "-1 1 -1 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03265_3():
    actual_output = problem_p03265("31 -41 -59 26")
    expected_output = "-126 -64 -36 -131"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
