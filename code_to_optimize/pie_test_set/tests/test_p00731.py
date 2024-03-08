from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00731_0():
    input_content = "6 6\n4 4 X X T T\n4 7 8 2 X 7\n3 X X X 1 8\n1 2 X X X 6\n1 1 2 4 4 7\nS S 2 3 X X\n2 10\nT 1\n1 X\n1 X\n1 X\n1 1\n1 X\n1 X\n1 1\n1 X\nS S\n2 10\nT X\n1 X\n1 X\n1 X\n1 1\n1 X\n1 X\n1 1\n1 X\nS S\n10 10\nT T T T T T T T T T\nX 2 X X X X X 3 4 X\n9 8 9 X X X 2 9 X 9\n7 7 X 7 3 X X 8 9 X\n8 9 9 9 6 3 X 5 X 5\n8 9 9 9 6 X X 5 X 5\n8 6 5 4 6 8 X 5 X 5\n8 9 3 9 6 8 X 5 X 5\n8 3 9 9 6 X X X 5 X\nS S S S S S S S S S\n10 7\n2 3 2 3 2 3 2 3 T T\n1 2 3 2 3 2 3 2 3 2\n3 2 3 2 3 2 3 2 3 4\n3 2 3 2 3 2 3 2 3 5\n3 2 3 1 3 2 3 2 3 5\n2 2 3 2 4 2 3 2 3 5\nS S 2 3 2 1 2 3 2 3\n0 0"
    expected_output = "12\n5\n-1\n22\n12"
    run_pie_test_case("../p00731.py", input_content, expected_output)


def test_problem_p00731_1():
    input_content = "6 6\n4 4 X X T T\n4 7 8 2 X 7\n3 X X X 1 8\n1 2 X X X 6\n1 1 2 4 4 7\nS S 2 3 X X\n2 10\nT 1\n1 X\n1 X\n1 X\n1 1\n1 X\n1 X\n1 1\n1 X\nS S\n2 10\nT X\n1 X\n1 X\n1 X\n1 1\n1 X\n1 X\n1 1\n1 X\nS S\n10 10\nT T T T T T T T T T\nX 2 X X X X X 3 4 X\n9 8 9 X X X 2 9 X 9\n7 7 X 7 3 X X 8 9 X\n8 9 9 9 6 3 X 5 X 5\n8 9 9 9 6 X X 5 X 5\n8 6 5 4 6 8 X 5 X 5\n8 9 3 9 6 8 X 5 X 5\n8 3 9 9 6 X X X 5 X\nS S S S S S S S S S\n10 7\n2 3 2 3 2 3 2 3 T T\n1 2 3 2 3 2 3 2 3 2\n3 2 3 2 3 2 3 2 3 4\n3 2 3 2 3 2 3 2 3 5\n3 2 3 1 3 2 3 2 3 5\n2 2 3 2 4 2 3 2 3 5\nS S 2 3 2 1 2 3 2 3\n0 0"
    expected_output = "12\n5\n-1\n22\n12"
    run_pie_test_case("../p00731.py", input_content, expected_output)
