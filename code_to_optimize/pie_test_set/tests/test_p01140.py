from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01140_0():
    input_content = "3 3\n1\n1\n4\n2\n3\n1\n1 2\n10\n10\n10\n0 0"
    expected_output = "6\n2"
    run_pie_test_case("../p01140.py", input_content, expected_output)


def test_problem_p01140_1():
    input_content = "3 3\n1\n1\n4\n2\n3\n1\n1 2\n10\n10\n10\n0 0"
    expected_output = "6\n2"
    run_pie_test_case("../p01140.py", input_content, expected_output)
