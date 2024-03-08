from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03389_0():
    input_content = "2 5 4"
    expected_output = "2"
    run_pie_test_case("../p03389.py", input_content, expected_output)
