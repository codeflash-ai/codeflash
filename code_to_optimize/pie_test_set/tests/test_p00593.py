from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00593_0():
    input_content = "3\n4\n0"
    expected_output = "Case 1:\n  1  2  6\n  3  5  7\n  4  8  9\nCase 2:\n  1  2  6  7\n  3  5  8 13\n  4  9 12 14\n 10 11 15 16"
    run_pie_test_case("../p00593.py", input_content, expected_output)


def test_problem_p00593_1():
    input_content = "3\n4\n0"
    expected_output = "Case 1:\n  1  2  6\n  3  5  7\n  4  8  9\nCase 2:\n  1  2  6  7\n  3  5  8 13\n  4  9 12 14\n 10 11 15 16"
    run_pie_test_case("../p00593.py", input_content, expected_output)
