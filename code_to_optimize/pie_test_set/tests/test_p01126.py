from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01126_0():
    input_content = "4 4 1\n3 1 2\n2 2 3\n3 3 4\n1 3 4\n0 0 0"
    expected_output = "4"
    run_pie_test_case("../p01126.py", input_content, expected_output)


def test_problem_p01126_1():
    input_content = "4 4 1\n3 1 2\n2 2 3\n3 3 4\n1 3 4\n0 0 0"
    expected_output = "4"
    run_pie_test_case("../p01126.py", input_content, expected_output)
