from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02859_0():
    input_content = "2"
    expected_output = "4"
    run_pie_test_case("../p02859.py", input_content, expected_output)
