from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04016_0():
    input_content = "87654\n30"
    expected_output = "10"
    run_pie_test_case("../p04016.py", input_content, expected_output)
