from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00180_0():
    input_content = "5 6\n0 2 1\n2 1 3\n2 3 8\n1 3 2\n3 4 5\n1 4 4\n3 3\n1 2 3\n2 0 3\n0 1 3\n0 0"
    expected_output = "10\n6"
    run_pie_test_case("../p00180.py", input_content, expected_output)


def test_problem_p00180_1():
    input_content = "5 6\n0 2 1\n2 1 3\n2 3 8\n1 3 2\n3 4 5\n1 4 4\n3 3\n1 2 3\n2 0 3\n0 1 3\n0 0"
    expected_output = "10\n6"
    run_pie_test_case("../p00180.py", input_content, expected_output)
