from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02911_0():
    input_content = "6 3 4\n3\n1\n3\n2"
    expected_output = "No\nNo\nYes\nNo\nNo\nNo"
    run_pie_test_case("../p02911.py", input_content, expected_output)


def test_problem_p02911_1():
    input_content = "6 5 4\n3\n1\n3\n2"
    expected_output = "Yes\nYes\nYes\nYes\nYes\nYes"
    run_pie_test_case("../p02911.py", input_content, expected_output)


def test_problem_p02911_2():
    input_content = "10 13 15\n3\n1\n4\n1\n5\n9\n2\n6\n5\n3\n5\n8\n9\n7\n9"
    expected_output = "No\nNo\nNo\nNo\nYes\nNo\nNo\nNo\nYes\nNo"
    run_pie_test_case("../p02911.py", input_content, expected_output)


def test_problem_p02911_3():
    input_content = "6 3 4\n3\n1\n3\n2"
    expected_output = "No\nNo\nYes\nNo\nNo\nNo"
    run_pie_test_case("../p02911.py", input_content, expected_output)
