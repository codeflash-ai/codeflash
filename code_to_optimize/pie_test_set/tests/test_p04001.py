from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04001_0():
    input_content = "125"
    expected_output = "176"
    run_pie_test_case("../p04001.py", input_content, expected_output)
