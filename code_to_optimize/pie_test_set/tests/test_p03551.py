from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03551_0():
    input_content = "1 1"
    expected_output = "3800"
    run_pie_test_case("../p03551.py", input_content, expected_output)
