from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04002_0():
    input_content = "4 5 8\n1 1\n1 4\n1 5\n2 3\n3 1\n3 2\n3 4\n4 4"
    expected_output = "0\n0\n0\n2\n4\n0\n0\n0\n0\n0"
    run_pie_test_case("../p04002.py", input_content, expected_output)
