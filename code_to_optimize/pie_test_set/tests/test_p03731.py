from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03731_0():
    input_content = "2 4\n0 3"
    expected_output = "7"
    run_pie_test_case("../p03731.py", input_content, expected_output)
