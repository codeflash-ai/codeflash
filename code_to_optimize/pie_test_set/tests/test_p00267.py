from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00267_0():
    input_content = "10\n4 9 1 9 5 9 2 3 2 1\n8 7 6 5 10 5 5 4 7 6\n5\n4 3 2 5 1\n4 4 4 4 4\n4\n4 1 3 2\n4 3 2 1\n0"
    expected_output = "3\n1\nNA"
    run_pie_test_case("../p00267.py", input_content, expected_output)


def test_problem_p00267_1():
    input_content = "10\n4 9 1 9 5 9 2 3 2 1\n8 7 6 5 10 5 5 4 7 6\n5\n4 3 2 5 1\n4 4 4 4 4\n4\n4 1 3 2\n4 3 2 1\n0"
    expected_output = "3\n1\nNA"
    run_pie_test_case("../p00267.py", input_content, expected_output)
