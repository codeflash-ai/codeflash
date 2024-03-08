from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03584_0():
    input_content = "3 5\n3 3\n4 4\n2 5"
    expected_output = "8"
    run_pie_test_case("../p03584.py", input_content, expected_output)
