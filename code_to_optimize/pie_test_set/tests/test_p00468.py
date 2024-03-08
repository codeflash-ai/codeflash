from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00468_0():
    input_content = "6\n5\n1 2\n1 3\n3 4\n2 3\n4 5\n6\n5\n2 3\n3 4\n4 5\n5 6\n2 5\n0\n0"
    expected_output = "3\n0"
    run_pie_test_case("../p00468.py", input_content, expected_output)


def test_problem_p00468_1():
    input_content = "6\n5\n1 2\n1 3\n3 4\n2 3\n4 5\n6\n5\n2 3\n3 4\n4 5\n5 6\n2 5\n0\n0"
    expected_output = "3\n0"
    run_pie_test_case("../p00468.py", input_content, expected_output)
