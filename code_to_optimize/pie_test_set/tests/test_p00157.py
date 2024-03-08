from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00157_0():
    input_content = "6\n1 1\n4 3\n6 5\n8 6\n10 10\n14 14\n5\n2 2\n5 4\n6 6\n9 8\n15 10\n4\n1 1\n4 3\n6 5\n8 6\n3\n2 2\n5 4\n6 6\n4\n1 1\n4 3\n6 5\n8 6\n4\n10 10\n12 11\n18 15\n24 20\n0"
    expected_output = "9\n6\n8"
    run_pie_test_case("../p00157.py", input_content, expected_output)


def test_problem_p00157_1():
    input_content = "6\n1 1\n4 3\n6 5\n8 6\n10 10\n14 14\n5\n2 2\n5 4\n6 6\n9 8\n15 10\n4\n1 1\n4 3\n6 5\n8 6\n3\n2 2\n5 4\n6 6\n4\n1 1\n4 3\n6 5\n8 6\n4\n10 10\n12 11\n18 15\n24 20\n0"
    expected_output = "9\n6\n8"
    run_pie_test_case("../p00157.py", input_content, expected_output)
