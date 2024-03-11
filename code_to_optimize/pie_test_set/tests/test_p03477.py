from code_to_optimize.pie_test_set.p03477 import problem_p03477


def test_problem_p03477_0():
    actual_output = problem_p03477("3 8 7 1")
    expected_output = "Left"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03477_1():
    actual_output = problem_p03477("3 8 7 1")
    expected_output = "Left"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03477_2():
    actual_output = problem_p03477("1 7 6 4")
    expected_output = "Right"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03477_3():
    actual_output = problem_p03477("3 4 5 2")
    expected_output = "Balanced"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
