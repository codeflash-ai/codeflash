from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03501_0():
    input_content = "7 17 120"
    expected_output = "119"
    run_pie_test_case("../p03501.py", input_content, expected_output)
