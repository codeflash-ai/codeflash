from code_to_optimize.pie_test_set.p03371 import problem_p03371


def test_problem_p03371_0():
    actual_output = problem_p03371("1500 2000 1600 3 2")
    expected_output = "7900"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03371_1():
    actual_output = problem_p03371("1500 2000 1600 3 2")
    expected_output = "7900"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03371_2():
    actual_output = problem_p03371("1500 2000 500 90000 100000")
    expected_output = "100000000"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03371_3():
    actual_output = problem_p03371("1500 2000 1900 3 2")
    expected_output = "8500"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
