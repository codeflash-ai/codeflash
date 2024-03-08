from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00033_0():
    input_content = "2\n3 1 4 2 5 6 7 8 9 10\n10 9 8 7 6 5 4 3 2 1"
    expected_output = "YES\nNO"
    run_pie_test_case("../p00033.py", input_content, expected_output)


def test_problem_p00033_1():
    input_content = "2\n3 1 4 2 5 6 7 8 9 10\n10 9 8 7 6 5 4 3 2 1"
    expected_output = "YES\nNO"
    run_pie_test_case("../p00033.py", input_content, expected_output)
