from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03800_0():
    input_content = "6\nooxoox"
    expected_output = "SSSWWS"
    run_pie_test_case("../p03800.py", input_content, expected_output)
