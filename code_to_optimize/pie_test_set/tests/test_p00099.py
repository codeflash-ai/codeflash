from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00099_0():
    input_content = "3 5\n1 4\n2 5\n1 3\n3 6\n2 7"
    expected_output = "1 4\n2 5\n1 7\n1 7\n2 12"
    run_pie_test_case("../p00099.py", input_content, expected_output)


def test_problem_p00099_1():
    input_content = "3 5\n1 4\n2 5\n1 3\n3 6\n2 7"
    expected_output = "1 4\n2 5\n1 7\n1 7\n2 12"
    run_pie_test_case("../p00099.py", input_content, expected_output)
