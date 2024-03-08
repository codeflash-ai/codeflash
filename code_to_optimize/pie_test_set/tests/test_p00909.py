from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00909_0():
    input_content = "2 2\n! 1 2 1\n? 1 2\n2 2\n! 1 2 1\n? 2 1\n4 7\n! 1 2 100\n? 2 3\n! 2 3 100\n? 2 3\n? 1 3\n! 4 3 150\n? 4 1\n0 0"
    expected_output = "1\n-1\nUNKNOWN\n100\n200\n-50"
    run_pie_test_case("../p00909.py", input_content, expected_output)


def test_problem_p00909_1():
    input_content = "2 2\n! 1 2 1\n? 1 2\n2 2\n! 1 2 1\n? 2 1\n4 7\n! 1 2 100\n? 2 3\n! 2 3 100\n? 2 3\n? 1 3\n! 4 3 150\n? 4 1\n0 0"
    expected_output = "1\n-1\nUNKNOWN\n100\n200\n-50"
    run_pie_test_case("../p00909.py", input_content, expected_output)
