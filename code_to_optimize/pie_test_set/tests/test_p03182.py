from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03182_0():
    input_content = "5 3\n1 3 10\n2 4 -10\n3 5 10"
    expected_output = "20"
    run_pie_test_case("../p03182.py", input_content, expected_output)
