from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03339_0():
    input_content = "5\nWEEWW"
    expected_output = "1"
    run_pie_test_case("../p03339.py", input_content, expected_output)
