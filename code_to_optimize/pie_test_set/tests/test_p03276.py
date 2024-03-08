from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03276_0():
    input_content = "5 3\n-30 -10 10 20 50"
    expected_output = "40"
    run_pie_test_case("../p03276.py", input_content, expected_output)
