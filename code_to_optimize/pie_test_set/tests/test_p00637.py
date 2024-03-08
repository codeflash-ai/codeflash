from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00637_0():
    input_content = "5\n1 2 3 5 6\n3\n7 8 9\n0"
    expected_output = "1-3 5-6\n7-9"
    run_pie_test_case("../p00637.py", input_content, expected_output)


def test_problem_p00637_1():
    input_content = "5\n1 2 3 5 6\n3\n7 8 9\n0"
    expected_output = "1-3 5-6\n7-9"
    run_pie_test_case("../p00637.py", input_content, expected_output)
