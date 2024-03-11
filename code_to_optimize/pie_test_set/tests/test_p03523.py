from code_to_optimize.pie_test_set.p03523 import problem_p03523


def test_problem_p03523_0():
    actual_output = problem_p03523("KIHBR")
    expected_output = "YES"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03523_1():
    actual_output = problem_p03523("AKIBAHARA")
    expected_output = "NO"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03523_2():
    actual_output = problem_p03523("KIHBR")
    expected_output = "YES"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03523_3():
    actual_output = problem_p03523("AAKIAHBAARA")
    expected_output = "NO"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
