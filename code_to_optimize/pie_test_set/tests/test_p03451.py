from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03451_0():
    input_content = "5\n3 2 2 4 1\n1 2 2 2 1"
    expected_output = "14"
    run_pie_test_case("../p03451.py", input_content, expected_output)
