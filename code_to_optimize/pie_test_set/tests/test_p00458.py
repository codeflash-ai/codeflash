from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00458_0():
    input_content = "3\n3\n1 1 0\n1 0 1\n1 1 0\n5\n3\n1 1 1 0 1\n1 1 0 0 0\n1 0 0 0 1\n0\n0"
    expected_output = "5\n5"
    run_pie_test_case("../p00458.py", input_content, expected_output)


def test_problem_p00458_1():
    input_content = "3\n3\n1 1 0\n1 0 1\n1 1 0\n5\n3\n1 1 1 0 1\n1 1 0 0 0\n1 0 0 0 1\n0\n0"
    expected_output = "5\n5"
    run_pie_test_case("../p00458.py", input_content, expected_output)
