from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01522_0():
    input_content = "6 2\n2 1 5\n4 2 3 4 6\n2\n1 2\n2 5"
    expected_output = "0"
    run_pie_test_case("../p01522.py", input_content, expected_output)
