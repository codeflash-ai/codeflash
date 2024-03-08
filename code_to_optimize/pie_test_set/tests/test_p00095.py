from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00095_0():
    input_content = "6\n1 14\n2 25\n3 42\n4 11\n5 40\n6 37"
    expected_output = "3 42"
    run_pie_test_case("../p00095.py", input_content, expected_output)


def test_problem_p00095_1():
    input_content = "6\n1 14\n2 25\n3 42\n4 11\n5 40\n6 37"
    expected_output = "3 42"
    run_pie_test_case("../p00095.py", input_content, expected_output)
