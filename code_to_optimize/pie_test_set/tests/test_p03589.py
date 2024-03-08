from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03589_0():
    input_content = "2"
    expected_output = "1 2 2"
    run_pie_test_case("../p03589.py", input_content, expected_output)
