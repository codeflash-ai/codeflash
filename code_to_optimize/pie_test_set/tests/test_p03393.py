from code_to_optimize.pie_test_set.p03393 import problem_p03393


def test_problem_p03393_0():
    actual_output = problem_p03393("atcoder")
    expected_output = "atcoderb"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03393_1():
    actual_output = problem_p03393("zyxwvutsrqponmlkjihgfedcba")
    expected_output = "-1"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03393_2():
    actual_output = problem_p03393("atcoder")
    expected_output = "atcoderb"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03393_3():
    actual_output = problem_p03393("abcdefghijklmnopqrstuvwzyx")
    expected_output = "abcdefghijklmnopqrstuvx"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03393_4():
    actual_output = problem_p03393("abc")
    expected_output = "abcd"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
