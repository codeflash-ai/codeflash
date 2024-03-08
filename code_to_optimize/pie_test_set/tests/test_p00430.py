from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00430_0():
    input_content = "5\n5\n0"
    expected_output = "5\n4 1\n3 2\n3 1 1\n2 2 1\n2 1 1 1\n1 1 1 1 1\n5\n4 1\n3 2\n3 1 1\n2 2 1\n2 1 1 1\n1 1 1 1 1"
    run_pie_test_case("../p00430.py", input_content, expected_output)


def test_problem_p00430_1():
    input_content = "5\n5\n0"
    expected_output = "5\n4 1\n3 2\n3 1 1\n2 2 1\n2 1 1 1\n1 1 1 1 1\n5\n4 1\n3 2\n3 1 1\n2 2 1\n2 1 1 1\n1 1 1 1 1"
    run_pie_test_case("../p00430.py", input_content, expected_output)
