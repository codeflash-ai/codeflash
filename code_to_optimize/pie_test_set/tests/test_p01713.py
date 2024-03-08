from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01713_0():
    input_content = "7\n2 0 -2 3 2 -2 0"
    expected_output = "4"
    run_pie_test_case("../p01713.py", input_content, expected_output)
