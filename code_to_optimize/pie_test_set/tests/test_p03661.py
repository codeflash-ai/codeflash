from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03661_0():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "1"
    run_pie_test_case("../p03661.py", input_content, expected_output)
