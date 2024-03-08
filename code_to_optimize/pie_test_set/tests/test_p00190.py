from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00190_0():
    input_content = "2\n1 0 3\n4 5 6 7 8\n9 0 11\n10\n0\n1 2 3\n4 5 6 7 8\n9 10 11\n0\n0\n11 10 9\n8 7 6 5 4\n3 2 1\n0\n-1"
    expected_output = "2\n0\nNA"
    run_pie_test_case("../p00190.py", input_content, expected_output)


def test_problem_p00190_1():
    input_content = "2\n1 0 3\n4 5 6 7 8\n9 0 11\n10\n0\n1 2 3\n4 5 6 7 8\n9 10 11\n0\n0\n11 10 9\n8 7 6 5 4\n3 2 1\n0\n-1"
    expected_output = "2\n0\nNA"
    run_pie_test_case("../p00190.py", input_content, expected_output)
