from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03319_0():
    input_content = "4 3\n2 3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03319.py", input_content, expected_output)
