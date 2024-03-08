from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02813_0():
    input_content = "3\n1 3 2\n3 1 2"
    expected_output = "3"
    run_pie_test_case("../p02813.py", input_content, expected_output)
