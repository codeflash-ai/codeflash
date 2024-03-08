from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00436_0():
    input_content = "2\n2\n1\n0"
    expected_output = "2\n4\n3\n1"
    run_pie_test_case("../p00436.py", input_content, expected_output)
