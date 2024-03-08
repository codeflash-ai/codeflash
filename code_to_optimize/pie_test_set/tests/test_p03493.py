from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03493_0():
    input_content = "101"
    expected_output = "2"
    run_pie_test_case("../p03493.py", input_content, expected_output)
