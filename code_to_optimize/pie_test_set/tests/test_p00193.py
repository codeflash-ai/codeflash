from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00193_0():
    input_content = "6 6\n6\n1 1\n6 1\n3 2\n3 5\n1 6\n5 6\n2\n1 3\n5 3\n6 6\n6\n3 2\n3 5\n6 1\n1 1\n1 6\n5 6\n2\n2 3\n5 3\n0 0"
    expected_output = "4\n4"
    run_pie_test_case("../p00193.py", input_content, expected_output)


def test_problem_p00193_1():
    input_content = "6 6\n6\n1 1\n6 1\n3 2\n3 5\n1 6\n5 6\n2\n1 3\n5 3\n6 6\n6\n3 2\n3 5\n6 1\n1 1\n1 6\n5 6\n2\n2 3\n5 3\n0 0"
    expected_output = "4\n4"
    run_pie_test_case("../p00193.py", input_content, expected_output)
