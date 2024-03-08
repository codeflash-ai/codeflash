from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00605_0():
    input_content = "2 3\n5 4 5\n1 2 3\n3 2 1\n3 5\n1 2 3 4 5\n0 1 0 1 2\n0 1 1 2 2\n1 0 3 1 1\n0 0"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p00605.py", input_content, expected_output)


def test_problem_p00605_1():
    input_content = "2 3\n5 4 5\n1 2 3\n3 2 1\n3 5\n1 2 3 4 5\n0 1 0 1 2\n0 1 1 2 2\n1 0 3 1 1\n0 0"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p00605.py", input_content, expected_output)
