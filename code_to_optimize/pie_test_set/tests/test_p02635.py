from code_to_optimize.pie_test_set.p02635 import problem_p02635


def test_problem_p02635_0():
    actual_output = problem_p02635("0101 1")
    expected_output = "4"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02635_1():
    actual_output = problem_p02635("0101 1")
    expected_output = "4"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02635_2():
    actual_output = problem_p02635("01100110 2")
    expected_output = "14"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p02635_3():
    actual_output = problem_p02635(
        "1101010010101101110111100011011111011000111101110101010010101010101 20"
    )
    expected_output = "113434815"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
