from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00144_0():
    input_content = "7\n1 4 2 5 4 3\n2 1 5\n3 1 6\n4 1 7\n5 2 7 6\n6 1 1\n7 0\n6\n1 2 2\n1 5 3\n1 2 1\n5 1 3\n6 3 3\n1 7 4"
    expected_output = "2\n2\nNA\n3\n3\n3"
    run_pie_test_case("../p00144.py", input_content, expected_output)


def test_problem_p00144_1():
    input_content = "7\n1 4 2 5 4 3\n2 1 5\n3 1 6\n4 1 7\n5 2 7 6\n6 1 1\n7 0\n6\n1 2 2\n1 5 3\n1 2 1\n5 1 3\n6 3 3\n1 7 4"
    expected_output = "2\n2\nNA\n3\n3\n3"
    run_pie_test_case("../p00144.py", input_content, expected_output)
