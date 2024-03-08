from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00441_0():
    input_content = "10\n9 4\n4 3\n1 1\n4 2\n2 4\n5 8\n4 0\n5 3\n0 5\n5 2\n10\n9 4\n4 3\n1 1\n4 2\n2 4\n5 8\n4 0\n5 3\n0 5\n5 2\n0"
    expected_output = "10\n10"
    run_pie_test_case("../p00441.py", input_content, expected_output)


def test_problem_p00441_1():
    input_content = "10\n9 4\n4 3\n1 1\n4 2\n2 4\n5 8\n4 0\n5 3\n0 5\n5 2\n10\n9 4\n4 3\n1 1\n4 2\n2 4\n5 8\n4 0\n5 3\n0 5\n5 2\n0"
    expected_output = "10\n10"
    run_pie_test_case("../p00441.py", input_content, expected_output)
