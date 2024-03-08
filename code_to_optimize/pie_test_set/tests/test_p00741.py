from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00741_0():
    input_content = "1 1\n0\n2 2\n0 1\n1 0\n3 2\n1 1 1\n1 1 1\n5 4\n1 0 1 0 0\n1 0 0 0 0\n1 0 1 0 1\n1 0 0 1 0\n5 4\n1 1 1 0 1\n1 0 1 0 1\n1 0 1 0 1\n1 0 1 1 1\n5 5\n1 0 1 0 1\n0 0 0 0 0\n1 0 1 0 1\n0 0 0 0 0\n1 0 1 0 1\n0 0"
    expected_output = "0\n1\n1\n3\n1\n9"
    run_pie_test_case("../p00741.py", input_content, expected_output)


def test_problem_p00741_1():
    input_content = "1 1\n0\n2 2\n0 1\n1 0\n3 2\n1 1 1\n1 1 1\n5 4\n1 0 1 0 0\n1 0 0 0 0\n1 0 1 0 1\n1 0 0 1 0\n5 4\n1 1 1 0 1\n1 0 1 0 1\n1 0 1 0 1\n1 0 1 1 1\n5 5\n1 0 1 0 1\n0 0 0 0 0\n1 0 1 0 1\n0 0 0 0 0\n1 0 1 0 1\n0 0"
    expected_output = "0\n1\n1\n3\n1\n9"
    run_pie_test_case("../p00741.py", input_content, expected_output)
