from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00464_0():
    input_content = "3 4 3\n1 0 1 1\n0 1 0 0\n1 0 1 0\n0 0 0"
    expected_output = "1 5"
    run_pie_test_case("../p00464.py", input_content, expected_output)


def test_problem_p00464_1():
    input_content = "3 4 3\n1 0 1 1\n0 1 0 0\n1 0 1 0\n0 0 0"
    expected_output = "1 5"
    run_pie_test_case("../p00464.py", input_content, expected_output)
