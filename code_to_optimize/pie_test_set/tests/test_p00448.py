from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00448_0():
    input_content = "2 5\n0 1 0 1 0\n1 0 0 0 1\n3 6\n1 0 0 0 1 0\n1 1 1 0 1 0\n1 0 1 1 0 1\n0 0"
    expected_output = "9\n15"
    run_pie_test_case("../p00448.py", input_content, expected_output)


def test_problem_p00448_1():
    input_content = "2 5\n0 1 0 1 0\n1 0 0 0 1\n3 6\n1 0 0 0 1 0\n1 1 1 0 1 0\n1 0 1 1 0 1\n0 0"
    expected_output = "9\n15"
    run_pie_test_case("../p00448.py", input_content, expected_output)
