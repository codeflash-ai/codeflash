from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03070_0():
    input_content = "4\n1\n1\n1\n2"
    expected_output = "18"
    run_pie_test_case("../p03070.py", input_content, expected_output)
