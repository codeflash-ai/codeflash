from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00706_0():
    input_content = "16\n10 8\n2 2\n2 5\n2 7\n3 3\n3 8\n4 2\n4 5\n4 8\n6 4\n6 7\n7 5\n7 8\n8 1\n8 4\n9 6\n10 3\n4 3\n8\n6 4\n1 2\n2 1\n2 4\n3 4\n4 2\n5 3\n6 1\n6 2\n3 2\n0"
    expected_output = "4\n3"
    run_pie_test_case("../p00706.py", input_content, expected_output)


def test_problem_p00706_1():
    input_content = "16\n10 8\n2 2\n2 5\n2 7\n3 3\n3 8\n4 2\n4 5\n4 8\n6 4\n6 7\n7 5\n7 8\n8 1\n8 4\n9 6\n10 3\n4 3\n8\n6 4\n1 2\n2 1\n2 4\n3 4\n4 2\n5 3\n6 1\n6 2\n3 2\n0"
    expected_output = "4\n3"
    run_pie_test_case("../p00706.py", input_content, expected_output)
