from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00433_0():
    input_content = "100 80 70 60\n80 70 80 90"
    expected_output = "320"
    run_pie_test_case("../p00433.py", input_content, expected_output)
