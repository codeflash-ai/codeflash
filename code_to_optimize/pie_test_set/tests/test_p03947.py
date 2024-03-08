from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03947_0():
    input_content = "BBBWW"
    expected_output = "1"
    run_pie_test_case("../p03947.py", input_content, expected_output)
