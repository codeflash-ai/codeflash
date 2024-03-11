from code_to_optimize.pie_test_set.p03860 import problem_p03860


def test_problem_p03860_0():
    actual_output = problem_p03860("AtCoder Beginner Contest")
    expected_output = "ABC"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03860_1():
    actual_output = problem_p03860("AtCoder Beginner Contest")
    expected_output = "ABC"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03860_2():
    actual_output = problem_p03860("AtCoder Snuke Contest")
    expected_output = "ASC"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03860_3():
    actual_output = problem_p03860("AtCoder X Contest")
    expected_output = "AXC"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
