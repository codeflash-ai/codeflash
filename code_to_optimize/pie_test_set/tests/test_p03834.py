from code_to_optimize.pie_test_set.p03834 import problem_p03834


def test_problem_p03834_0():
    actual_output = problem_p03834("happy,newyear,enjoy")
    expected_output = "happy newyear enjoy"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03834_1():
    actual_output = problem_p03834("haiku,atcoder,tasks")
    expected_output = "haiku atcoder tasks"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03834_2():
    actual_output = problem_p03834("abcde,fghihgf,edcba")
    expected_output = "abcde fghihgf edcba"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03834_3():
    actual_output = problem_p03834("happy,newyear,enjoy")
    expected_output = "happy newyear enjoy"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
