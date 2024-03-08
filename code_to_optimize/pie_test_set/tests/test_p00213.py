from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00213_0():
    input_content = "5 4 6\n1 6\n2 4\n3 3\n4 4\n5 1\n6 2\n0 0 1 0 0\n0 0 0 2 0\n0 0 0 3 0\n0 4 5 0 6\n3 3 1\n1 9\n0 0 1\n0 0 0\n0 0 0\n4 4 4\n1 6\n2 2\n3 4\n4 4\n0 1 0 0\n0 0 0 2\n0 0 3 0\n0 4 0 0\n0 0 0"
    expected_output = "1 1 1 2 2\n1 1 1 2 2\n4 4 3 3 3\n4 4 5 6 6\n1 1 1\n1 1 1\n1 1 1\nNA"
    run_pie_test_case("../p00213.py", input_content, expected_output)


def test_problem_p00213_1():
    input_content = "5 4 6\n1 6\n2 4\n3 3\n4 4\n5 1\n6 2\n0 0 1 0 0\n0 0 0 2 0\n0 0 0 3 0\n0 4 5 0 6\n3 3 1\n1 9\n0 0 1\n0 0 0\n0 0 0\n4 4 4\n1 6\n2 2\n3 4\n4 4\n0 1 0 0\n0 0 0 2\n0 0 3 0\n0 4 0 0\n0 0 0"
    expected_output = "1 1 1 2 2\n1 1 1 2 2\n4 4 3 3 3\n4 4 5 6 6\n1 1 1\n1 1 1\n1 1 1\nNA"
    run_pie_test_case("../p00213.py", input_content, expected_output)
