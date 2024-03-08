from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00725_0():
    input_content = "2 1\n3 2\n6 6\n1 0 0 2 1 0\n1 1 0 0 0 0\n0 0 0 0 0 3\n0 0 0 0 0 0\n1 0 0 0 0 1\n0 1 1 1 1 1\n6 1\n1 1 2 1 1 3\n6 1\n1 0 2 1 1 3\n12 1\n2 0 1 1 1 1 1 1 1 1 1 3\n13 1\n2 0 1 1 1 1 1 1 1 1 1 1 3\n0 0"
    expected_output = "1\n4\n-1\n4\n10\n-1"
    run_pie_test_case("../p00725.py", input_content, expected_output)


def test_problem_p00725_1():
    input_content = "2 1\n3 2\n6 6\n1 0 0 2 1 0\n1 1 0 0 0 0\n0 0 0 0 0 3\n0 0 0 0 0 0\n1 0 0 0 0 1\n0 1 1 1 1 1\n6 1\n1 1 2 1 1 3\n6 1\n1 0 2 1 1 3\n12 1\n2 0 1 1 1 1 1 1 1 1 1 3\n13 1\n2 0 1 1 1 1 1 1 1 1 1 1 3\n0 0"
    expected_output = "1\n4\n-1\n4\n10\n-1"
    run_pie_test_case("../p00725.py", input_content, expected_output)
