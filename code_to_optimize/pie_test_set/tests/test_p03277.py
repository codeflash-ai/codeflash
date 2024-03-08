from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03277_0():
    input_content = "3\n10 30 20"
    expected_output = "30"
    run_pie_test_case("../p03277.py", input_content, expected_output)
