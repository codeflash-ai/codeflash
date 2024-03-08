from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00047_0():
    input_content = "B,C\nA,C\nC,B\nA,B\nC,B"
    expected_output = "A"
    run_pie_test_case("../p00047.py", input_content, expected_output)
