from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00874_0():
    input_content = "5 5\n1 2 3 4 5\n1 2 3 4 5\n5 5\n2 5 4 1 3\n4 1 5 3 2\n5 5\n1 2 3 4 5\n3 3 3 4 5\n3 3\n7 7 7\n7 7 7\n3 3\n4 4 4\n4 3 4\n4 3\n4 2 2 4\n4 2 1\n4 4\n2 8 8 8\n2 3 8 3\n10 10\n9 9 9 9 9 9 9 9 9 9\n9 9 9 9 9 9 9 9 9 9\n10 9\n20 1 20 20 20 20 20 18 20 20\n20 20 20 20 7 20 20 20 20\n0 0"
    expected_output = "15\n15\n21\n21\n15\n13\n32\n90\n186"
    run_pie_test_case("../p00874.py", input_content, expected_output)


def test_problem_p00874_1():
    input_content = "5 5\n1 2 3 4 5\n1 2 3 4 5\n5 5\n2 5 4 1 3\n4 1 5 3 2\n5 5\n1 2 3 4 5\n3 3 3 4 5\n3 3\n7 7 7\n7 7 7\n3 3\n4 4 4\n4 3 4\n4 3\n4 2 2 4\n4 2 1\n4 4\n2 8 8 8\n2 3 8 3\n10 10\n9 9 9 9 9 9 9 9 9 9\n9 9 9 9 9 9 9 9 9 9\n10 9\n20 1 20 20 20 20 20 18 20 20\n20 20 20 20 7 20 20 20 20\n0 0"
    expected_output = "15\n15\n21\n21\n15\n13\n32\n90\n186"
    run_pie_test_case("../p00874.py", input_content, expected_output)
