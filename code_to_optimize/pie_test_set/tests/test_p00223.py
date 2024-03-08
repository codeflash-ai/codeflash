from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00223_0():
    input_content = "6 6\n2 4\n6 2\n0 0 0 0 1 0\n0 1 0 0 0 0\n0 1 0 0 0 0\n0 0 0 1 0 0\n0 0 0 0 0 1\n0 0 0 0 0 0\n3 3\n1 1\n3 3\n0 0 0\n0 1 0\n0 0 0\n0 0"
    expected_output = "3\nNA"
    run_pie_test_case("../p00223.py", input_content, expected_output)


def test_problem_p00223_1():
    input_content = "6 6\n2 4\n6 2\n0 0 0 0 1 0\n0 1 0 0 0 0\n0 1 0 0 0 0\n0 0 0 1 0 0\n0 0 0 0 0 1\n0 0 0 0 0 0\n3 3\n1 1\n3 3\n0 0 0\n0 1 0\n0 0 0\n0 0"
    expected_output = "3\nNA"
    run_pie_test_case("../p00223.py", input_content, expected_output)
