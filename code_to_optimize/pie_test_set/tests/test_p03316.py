from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03316_0():
    input_content = "12"
    expected_output = "Yes"
    run_pie_test_case("../p03316.py", input_content, expected_output)
