from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03524_0():
    input_content = "abac"
    expected_output = "YES"
    run_pie_test_case("../p03524.py", input_content, expected_output)
