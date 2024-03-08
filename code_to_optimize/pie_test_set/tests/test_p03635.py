from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03635_0():
    input_content = "3 4"
    expected_output = "6"
    run_pie_test_case("../p03635.py", input_content, expected_output)
