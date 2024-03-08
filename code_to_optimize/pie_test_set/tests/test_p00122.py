from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00122_0():
    input_content = "6 1\n10\n6 4 3 3 1 2 0 5 4 6 1 8 5 9 7 7 8 6 8 3\n6 1\n10\n6 4 3 3 1 2 0 5 4 6 1 8 5 9 7 7 8 6 9 0\n0 0"
    expected_output = "OK\nNA"
    run_pie_test_case("../p00122.py", input_content, expected_output)


def test_problem_p00122_1():
    input_content = "6 1\n10\n6 4 3 3 1 2 0 5 4 6 1 8 5 9 7 7 8 6 8 3\n6 1\n10\n6 4 3 3 1 2 0 5 4 6 1 8 5 9 7 7 8 6 9 0\n0 0"
    expected_output = "OK\nNA"
    run_pie_test_case("../p00122.py", input_content, expected_output)
