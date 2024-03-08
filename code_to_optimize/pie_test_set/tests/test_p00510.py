from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00510_0():
    input_content = "3\n2\n2 3\n2 3\n4 1"
    expected_output = "3"
    run_pie_test_case("../p00510.py", input_content, expected_output)
