from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03782_0():
    input_content = "3 6\n1 4 3"
    expected_output = "1"
    run_pie_test_case("../p03782.py", input_content, expected_output)
