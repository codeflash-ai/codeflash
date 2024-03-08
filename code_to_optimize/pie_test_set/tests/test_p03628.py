from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03628_0():
    input_content = "3\naab\nccb"
    expected_output = "6"
    run_pie_test_case("../p03628.py", input_content, expected_output)
