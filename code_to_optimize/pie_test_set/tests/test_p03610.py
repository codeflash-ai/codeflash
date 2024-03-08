from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03610_0():
    input_content = "atcoder"
    expected_output = "acdr"
    run_pie_test_case("../p03610.py", input_content, expected_output)
