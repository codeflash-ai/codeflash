from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02393_0():
    input_content = "3 8 1"
    expected_output = "1 3 8"
    run_pie_test_case("../p02393.py", input_content, expected_output)
