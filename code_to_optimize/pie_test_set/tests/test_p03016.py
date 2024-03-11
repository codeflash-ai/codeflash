from code_to_optimize.pie_test_set.p03016 import problem_p03016


def test_problem_p03016_0():
    actual_output = problem_p03016("5 3 4 10007")
    expected_output = "5563"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03016_1():
    actual_output = problem_p03016("4 8 1 1000000")
    expected_output = "891011"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03016_2():
    actual_output = problem_p03016("107 10000000000007 1000000000000007 998244353")
    expected_output = "39122908"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03016_3():
    actual_output = problem_p03016("5 3 4 10007")
    expected_output = "5563"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
