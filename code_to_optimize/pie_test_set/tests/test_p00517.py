from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00517_0():
    input_content = "4 3 3\n1 1\n3 3\n4 1"
    expected_output = "5"
    run_pie_test_case("../p00517.py", input_content, expected_output)
