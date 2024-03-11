from code_to_optimize.pie_test_set.p03469 import problem_p03469


def test_problem_p03469_0():
    actual_output = problem_p03469("2017/01/07")
    expected_output = "2018/01/07"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03469_1():
    actual_output = problem_p03469("2017/01/07")
    expected_output = "2018/01/07"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output


def test_problem_p03469_2():
    actual_output = problem_p03469("2017/01/31")
    expected_output = "2018/01/31"
    if isinstance(actual_output, type(expected_output)):
        assert actual_output == expected_output
    else:
        # Cast expected output to the type of actual output if they differ
        cast_expected_output = type(actual_output)(expected_output)
        assert actual_output == cast_expected_output
