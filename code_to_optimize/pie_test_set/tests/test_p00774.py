from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00774_0():
    input_content = "1\n6 9 9 9 9\n5\n5 9 5 5 9\n5 5 6 9 9\n4 6 3 6 9\n3 3 2 9 9\n2 2 1 1 1\n10\n3 5 6 5 6\n2 2 2 8 3\n6 2 5 9 2\n7 7 7 6 1\n4 6 6 4 9\n8 9 1 1 8\n5 6 1 8 1\n6 8 2 1 2\n9 6 3 3 5\n5 3 8 8 8\n5\n1 2 3 4 5\n6 7 8 9 1\n2 3 4 5 6\n7 8 9 1 2\n3 4 5 6 7\n3\n2 2 8 7 4\n6 5 7 7 7\n8 8 9 9 9\n0"
    expected_output = "36\n38\n99\n0\n72"
    run_pie_test_case("../p00774.py", input_content, expected_output)


def test_problem_p00774_1():
    input_content = "1\n6 9 9 9 9\n5\n5 9 5 5 9\n5 5 6 9 9\n4 6 3 6 9\n3 3 2 9 9\n2 2 1 1 1\n10\n3 5 6 5 6\n2 2 2 8 3\n6 2 5 9 2\n7 7 7 6 1\n4 6 6 4 9\n8 9 1 1 8\n5 6 1 8 1\n6 8 2 1 2\n9 6 3 3 5\n5 3 8 8 8\n5\n1 2 3 4 5\n6 7 8 9 1\n2 3 4 5 6\n7 8 9 1 2\n3 4 5 6 7\n3\n2 2 8 7 4\n6 5 7 7 7\n8 8 9 9 9\n0"
    expected_output = "36\n38\n99\n0\n72"
    run_pie_test_case("../p00774.py", input_content, expected_output)
