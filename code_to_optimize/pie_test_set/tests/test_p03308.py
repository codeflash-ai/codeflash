from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03308_0():
    input_content = "4\n1 4 6 3"
    expected_output = "5"
    run_pie_test_case("../p03308.py", input_content, expected_output)
