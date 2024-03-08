from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03795_0():
    input_content = "20"
    expected_output = "15800"
    run_pie_test_case("../p03795.py", input_content, expected_output)
