from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00143_0():
    input_content = "5\n2 5 9 2 8 9 2 11 6 5\n2 5 9 2 8 9 2 11 12 6\n2 5 9 2 8 9 2 11 11 9\n14 1 25 7 17 12 17 9 20 5\n14 1 25 7 17 12 22 13 20 5"
    expected_output = "OK\nNG\nNG\nNG\nOK"
    run_pie_test_case("../p00143.py", input_content, expected_output)


def test_problem_p00143_1():
    input_content = "5\n2 5 9 2 8 9 2 11 6 5\n2 5 9 2 8 9 2 11 12 6\n2 5 9 2 8 9 2 11 11 9\n14 1 25 7 17 12 17 9 20 5\n14 1 25 7 17 12 22 13 20 5"
    expected_output = "OK\nNG\nNG\nNG\nOK"
    run_pie_test_case("../p00143.py", input_content, expected_output)
