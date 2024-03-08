from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00209_0():
    input_content = "8 4\n2 1 3 1 1 5 1 3\n2 3 2 4 1 0 2 1\n0 3 1 2 1 1 4 2\n1 2 3 2 1 1 5 4\n0 2 0 1 1 3 2 1\n1 3 1 2 2 4 3 2\n5 1 2 1 4 1 1 5\n4 1 1 0 1 2 2 1\n2 -1 -1 -1\n0 3 -1 -1\n-1 2 2 4\n-1 1 -1  1\n5 3\n1 0 2 3 5\n2 3 7 2 1\n2 5 4 2 2\n8 9 0 3 3\n3 6 0 4 7\n-1 -1 2\n-1 3 5\n0 4 -1\n0 0"
    expected_output = "4 2\nNA"
    run_pie_test_case("../p00209.py", input_content, expected_output)


def test_problem_p00209_1():
    input_content = "8 4\n2 1 3 1 1 5 1 3\n2 3 2 4 1 0 2 1\n0 3 1 2 1 1 4 2\n1 2 3 2 1 1 5 4\n0 2 0 1 1 3 2 1\n1 3 1 2 2 4 3 2\n5 1 2 1 4 1 1 5\n4 1 1 0 1 2 2 1\n2 -1 -1 -1\n0 3 -1 -1\n-1 2 2 4\n-1 1 -1  1\n5 3\n1 0 2 3 5\n2 3 7 2 1\n2 5 4 2 2\n8 9 0 3 3\n3 6 0 4 7\n-1 -1 2\n-1 3 5\n0 4 -1\n0 0"
    expected_output = "4 2\nNA"
    run_pie_test_case("../p00209.py", input_content, expected_output)
