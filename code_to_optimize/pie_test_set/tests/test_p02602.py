from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02602_0():
    input_content = "5 3\n96 98 95 100 20"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p02602.py", input_content, expected_output)


def test_problem_p02602_1():
    input_content = "3 2\n1001 869120 1001"
    expected_output = "No"
    run_pie_test_case("../p02602.py", input_content, expected_output)


def test_problem_p02602_2():
    input_content = "15 7\n3 1 4 1 5 9 2 6 5 3 5 8 9 7 9"
    expected_output = "Yes\nYes\nNo\nYes\nYes\nNo\nYes\nYes"
    run_pie_test_case("../p02602.py", input_content, expected_output)


def test_problem_p02602_3():
    input_content = "5 3\n96 98 95 100 20"
    expected_output = "Yes\nNo"
    run_pie_test_case("../p02602.py", input_content, expected_output)
