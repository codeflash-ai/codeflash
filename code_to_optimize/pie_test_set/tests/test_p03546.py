from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03546_0():
    input_content = "2 4\n0 9 9 9 9 9 9 9 9 9\n9 0 9 9 9 9 9 9 9 9\n9 9 0 9 9 9 9 9 9 9\n9 9 9 0 9 9 9 9 9 9\n9 9 9 9 0 9 9 9 9 2\n9 9 9 9 9 0 9 9 9 9\n9 9 9 9 9 9 0 9 9 9\n9 9 9 9 9 9 9 0 9 9\n9 9 9 9 2 9 9 9 0 9\n9 2 9 9 9 9 9 9 9 0\n-1 -1 -1 -1\n8 1 1 8"
    expected_output = "12"
    run_pie_test_case("../p03546.py", input_content, expected_output)


def test_problem_p03546_1():
    input_content = "2 4\n0 9 9 9 9 9 9 9 9 9\n9 0 9 9 9 9 9 9 9 9\n9 9 0 9 9 9 9 9 9 9\n9 9 9 0 9 9 9 9 9 9\n9 9 9 9 0 9 9 9 9 2\n9 9 9 9 9 0 9 9 9 9\n9 9 9 9 9 9 0 9 9 9\n9 9 9 9 9 9 9 0 9 9\n9 9 9 9 2 9 9 9 0 9\n9 2 9 9 9 9 9 9 9 0\n-1 -1 -1 -1\n8 1 1 8"
    expected_output = "12"
    run_pie_test_case("../p03546.py", input_content, expected_output)


def test_problem_p03546_2():
    input_content = "3 5\n0 4 3 6 2 7 2 5 3 3\n4 0 5 3 7 5 3 7 2 7\n5 7 0 7 2 9 3 2 9 1\n3 6 2 0 2 4 6 4 2 3\n3 5 7 4 0 6 9 7 6 7\n9 8 5 2 2 0 4 7 6 5\n5 4 6 3 2 3 0 5 4 3\n3 6 2 3 4 2 4 0 8 9\n4 6 5 4 3 5 3 2 0 8\n2 1 3 4 5 7 8 6 4 0\n3 5 2 6 1\n2 5 3 2 1\n6 9 2 5 6"
    expected_output = "47"
    run_pie_test_case("../p03546.py", input_content, expected_output)


def test_problem_p03546_3():
    input_content = "5 5\n0 999 999 999 999 999 999 999 999 999\n999 0 999 999 999 999 999 999 999 999\n999 999 0 999 999 999 999 999 999 999\n999 999 999 0 999 999 999 999 999 999\n999 999 999 999 0 999 999 999 999 999\n999 999 999 999 999 0 999 999 999 999\n999 999 999 999 999 999 0 999 999 999\n999 999 999 999 999 999 999 0 999 999\n999 999 999 999 999 999 999 999 0 999\n999 999 999 999 999 999 999 999 999 0\n1 1 1 1 1\n1 1 1 1 1\n1 1 1 1 1\n1 1 1 1 1\n1 1 1 1 1"
    expected_output = "0"
    run_pie_test_case("../p03546.py", input_content, expected_output)
