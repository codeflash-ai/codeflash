from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00086_0():
    input_content = "1 3\n3 4\n3 5\n3 6\n4 6\n4 7\n4 7\n5 6\n6 7\n5 8\n5 8\n6 8\n6 9\n7 9\n8 9\n9 2\n0 0\n1 3\n3 4\n3 4\n4 2\n0 0"
    expected_output = "OK\nNG"
    run_pie_test_case("../p00086.py", input_content, expected_output)


def test_problem_p00086_1():
    input_content = "1 3\n3 4\n3 5\n3 6\n4 6\n4 7\n4 7\n5 6\n6 7\n5 8\n5 8\n6 8\n6 9\n7 9\n8 9\n9 2\n0 0\n1 3\n3 4\n3 4\n4 2\n0 0"
    expected_output = "OK\nNG"
    run_pie_test_case("../p00086.py", input_content, expected_output)
