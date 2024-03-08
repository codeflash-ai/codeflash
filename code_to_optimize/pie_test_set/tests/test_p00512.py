from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00512_0():
    input_content = "5\nA 20\nB 20\nA 20\nAB 10\nZ 10"
    expected_output = "A 40\nB 20\nZ 10\nAB 10"
    run_pie_test_case("../p00512.py", input_content, expected_output)
