from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02441_0():
    input_content = "9\n1 4 1 4 2 1 3 5 6\n3\n0 9 1\n1 6 1\n3 7 5"
    expected_output = "3\n2\n0"
    run_pie_test_case("../p02441.py", input_content, expected_output)


def test_problem_p02441_1():
    input_content = "9\n1 4 1 4 2 1 3 5 6\n3\n0 9 1\n1 6 1\n3 7 5"
    expected_output = "3\n2\n0"
    run_pie_test_case("../p02441.py", input_content, expected_output)
