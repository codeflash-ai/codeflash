from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01520_0():
    input_content = "2 10 2\n3 4"
    expected_output = "2"
    run_pie_test_case("../p01520.py", input_content, expected_output)
