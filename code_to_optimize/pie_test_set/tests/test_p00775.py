from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00775_0():
    input_content = "2 3\n-2 -1 3\n0 1 3\n2 3 3\n2 2\n-2 0 4\n0 2 3\n2 6\n-3 3 1\n-2 3 2\n-1 3 3\n0 3 4\n1 3 5\n2 3 6\n2 6\n-3 3 1\n-3 2 2\n-3 1 3\n-3 0 4\n-3 -1 5\n-3 -2 6\n0 0"
    expected_output = "0.0000\n3.0000\n2.2679\n2.2679"
    run_pie_test_case("../p00775.py", input_content, expected_output)


def test_problem_p00775_1():
    input_content = "2 3\n-2 -1 3\n0 1 3\n2 3 3\n2 2\n-2 0 4\n0 2 3\n2 6\n-3 3 1\n-2 3 2\n-1 3 3\n0 3 4\n1 3 5\n2 3 6\n2 6\n-3 3 1\n-3 2 2\n-3 1 3\n-3 0 4\n-3 -1 5\n-3 -2 6\n0 0"
    expected_output = "0.0000\n3.0000\n2.2679\n2.2679"
    run_pie_test_case("../p00775.py", input_content, expected_output)
