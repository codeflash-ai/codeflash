from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00292_0():
    input_content = "3\n10 3\n2 10\n4 2"
    expected_output = "1\n2\n2"
    run_pie_test_case("../p00292.py", input_content, expected_output)
