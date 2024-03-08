from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02267_0():
    input_content = "5\n1 2 3 4 5\n3\n3 4 1"
    expected_output = "3"
    run_pie_test_case("../p02267.py", input_content, expected_output)
