from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04013_0():
    input_content = "4 8\n7 9 8 9"
    expected_output = "5"
    run_pie_test_case("../p04013.py", input_content, expected_output)
