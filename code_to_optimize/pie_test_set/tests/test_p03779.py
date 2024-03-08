from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03779_0():
    input_content = "6"
    expected_output = "3"
    run_pie_test_case("../p03779.py", input_content, expected_output)
