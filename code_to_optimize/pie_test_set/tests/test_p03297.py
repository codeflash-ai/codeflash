from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03297_0():
    input_content = "14\n9 7 5 9\n9 7 6 9\n14 10 7 12\n14 10 8 12\n14 10 9 12\n14 10 7 11\n14 10 8 11\n14 10 9 11\n9 10 5 10\n10 10 5 10\n11 10 5 10\n16 10 5 10\n1000000000000000000 17 14 999999999999999985\n1000000000000000000 17 15 999999999999999985"
    expected_output = "No\nYes\nNo\nYes\nYes\nNo\nNo\nYes\nNo\nYes\nYes\nNo\nNo\nYes"
    run_pie_test_case("../p03297.py", input_content, expected_output)


def test_problem_p03297_1():
    input_content = "24\n1 2 3 4\n1 2 4 3\n1 3 2 4\n1 3 4 2\n1 4 2 3\n1 4 3 2\n2 1 3 4\n2 1 4 3\n2 3 1 4\n2 3 4 1\n2 4 1 3\n2 4 3 1\n3 1 2 4\n3 1 4 2\n3 2 1 4\n3 2 4 1\n3 4 1 2\n3 4 2 1\n4 1 2 3\n4 1 3 2\n4 2 1 3\n4 2 3 1\n4 3 1 2\n4 3 2 1"
    expected_output = "No\nNo\nNo\nNo\nNo\nNo\nYes\nYes\nNo\nNo\nNo\nNo\nYes\nYes\nYes\nNo\nNo\nNo\nYes\nYes\nYes\nNo\nNo\nNo"
    run_pie_test_case("../p03297.py", input_content, expected_output)


def test_problem_p03297_2():
    input_content = "14\n9 7 5 9\n9 7 6 9\n14 10 7 12\n14 10 8 12\n14 10 9 12\n14 10 7 11\n14 10 8 11\n14 10 9 11\n9 10 5 10\n10 10 5 10\n11 10 5 10\n16 10 5 10\n1000000000000000000 17 14 999999999999999985\n1000000000000000000 17 15 999999999999999985"
    expected_output = "No\nYes\nNo\nYes\nYes\nNo\nNo\nYes\nNo\nYes\nYes\nNo\nNo\nYes"
    run_pie_test_case("../p03297.py", input_content, expected_output)
