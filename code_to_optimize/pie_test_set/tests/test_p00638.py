from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00638_0():
    input_content = "3\n2 3\n3 6\n1 2\n3\n2 3\n3 5\n1 2\n0"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p00638.py", input_content, expected_output)


def test_problem_p00638_1():
    input_content = "3\n2 3\n3 6\n1 2\n3\n2 3\n3 5\n1 2\n0"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p00638.py", input_content, expected_output)
