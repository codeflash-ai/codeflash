from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01143_0():
    input_content = "3 2 50\n1\n2\n3\n4 4 75\n1\n2\n3\n0\n3 1 10\n8\n1\n1\n0 0 0"
    expected_output = "150\n0\n112"
    run_pie_test_case("../p01143.py", input_content, expected_output)


def test_problem_p01143_1():
    input_content = "3 2 50\n1\n2\n3\n4 4 75\n1\n2\n3\n0\n3 1 10\n8\n1\n1\n0 0 0"
    expected_output = "150\n0\n112"
    run_pie_test_case("../p01143.py", input_content, expected_output)
