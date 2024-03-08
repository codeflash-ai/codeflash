from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03627_0():
    input_content = "6\n3 1 2 4 2 1"
    expected_output = "2"
    run_pie_test_case("../p03627.py", input_content, expected_output)
