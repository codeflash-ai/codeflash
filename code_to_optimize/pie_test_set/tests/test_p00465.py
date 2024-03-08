from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00465_0():
    input_content = "5\n2 2 1 2\n9 5\n1 17\n3 2 2 1\n6 1 20\n8 18 3\n8\n5 4 1 3\n5 5 4 5 5\n8 2 1 9 7\n1 1 3 5 1\n7 2 7 1 3\n6 5 6 2\n2 3 5 8 2 7\n1 6 9 4 5 1\n2 4 5 4 2 2\n5 4 2 5 3 3\n7 1 5 1 5 6\n6\n3 3 2 2\n2 9 2\n9 1 9\n2 9 2\n2 2 1 1\n1 3\n5 7\n0"
    expected_output = "15\n4\n9"
    run_pie_test_case("../p00465.py", input_content, expected_output)


def test_problem_p00465_1():
    input_content = "5\n2 2 1 2\n9 5\n1 17\n3 2 2 1\n6 1 20\n8 18 3\n8\n5 4 1 3\n5 5 4 5 5\n8 2 1 9 7\n1 1 3 5 1\n7 2 7 1 3\n6 5 6 2\n2 3 5 8 2 7\n1 6 9 4 5 1\n2 4 5 4 2 2\n5 4 2 5 3 3\n7 1 5 1 5 6\n6\n3 3 2 2\n2 9 2\n9 1 9\n2 9 2\n2 2 1 1\n1 3\n5 7\n0"
    expected_output = "15\n4\n9"
    run_pie_test_case("../p00465.py", input_content, expected_output)
