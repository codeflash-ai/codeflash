from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03701_0():
    input_content = "3\n5\n10\n15"
    expected_output = "25"
    run_pie_test_case("../p03701.py", input_content, expected_output)
