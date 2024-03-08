from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00481_0():
    input_content = "3 3 1\nS..\n...\n..1"
    expected_output = "4"
    run_pie_test_case("../p00481.py", input_content, expected_output)
