from code_to_optimize.pie_test_set.p02995 import problem_p02995


def test_problem_p02995_0():
    actual_output = problem_p02995("4 9 2 3")
    expected_output = "2"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02995_1():
    actual_output = problem_p02995("314159265358979323 846264338327950288 419716939 937510582")
    expected_output = "532105071133627368"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02995_2():
    actual_output = problem_p02995("10 40 6 8")
    expected_output = "23"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02995_3():
    actual_output = problem_p02995("4 9 2 3")
    expected_output = "2"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
