from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00454_0():
    input_content = "15 6\n10\n1 4 5 6\n2 1 4 5\n1 0 5 1\n6 1 7 5\n7 5 9 6\n7 0 9 2\n9 1 10 5\n11 0 14 1\n12 1 13 5\n11 5 14 6\n0 0"
    expected_output = "5"
    run_pie_test_case("../p00454.py", input_content, expected_output)


def test_problem_p00454_1():
    input_content = "15 6\n10\n1 4 5 6\n2 1 4 5\n1 0 5 1\n6 1 7 5\n7 5 9 6\n7 0 9 2\n9 1 10 5\n11 0 14 1\n12 1 13 5\n11 5 14 6\n0 0"
    expected_output = "5"
    run_pie_test_case("../p00454.py", input_content, expected_output)
