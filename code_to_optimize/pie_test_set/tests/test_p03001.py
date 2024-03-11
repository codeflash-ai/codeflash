from code_to_optimize.pie_test_set.p03001 import problem_p03001


def test_problem_p03001_0():
    actual_output = problem_p03001("2 3 1 2")
    expected_output = "3.000000 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03001_1():
    actual_output = problem_p03001("2 3 1 2")
    expected_output = "3.000000 0"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03001_2():
    actual_output = problem_p03001("2 2 1 1")
    expected_output = "2.000000 1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
