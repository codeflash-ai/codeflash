from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01334_0():
    input_content = "1\n0 0\n2\n1 1 0 1\n1 0 0 0\n2\n1 1 0 1\n1 1 1 0\n3\n0 1 2 2 2 1\n0 2 1 2 2 1\n0 0 0 1 1 1\n4\n3 2 2 0 3 2 2 1\n1 1 0 3 1 1 3 1\n0 3 2 3 3 0 2 3\n1 1 1 1 3 2 1 3\n0"
    expected_output = "1\n2\n1\n2\n3"
    run_pie_test_case("../p01334.py", input_content, expected_output)


def test_problem_p01334_1():
    input_content = "1\n0 0\n2\n1 1 0 1\n1 0 0 0\n2\n1 1 0 1\n1 1 1 0\n3\n0 1 2 2 2 1\n0 2 1 2 2 1\n0 0 0 1 1 1\n4\n3 2 2 0 3 2 2 1\n1 1 0 3 1 1 3 1\n0 3 2 3 3 0 2 3\n1 1 1 1 3 2 1 3\n0"
    expected_output = "1\n2\n1\n2\n3"
    run_pie_test_case("../p01334.py", input_content, expected_output)
