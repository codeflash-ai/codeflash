from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03418_0():
    input_content = "5 2"
    expected_output = "7"
    run_pie_test_case("../p03418.py", input_content, expected_output)
