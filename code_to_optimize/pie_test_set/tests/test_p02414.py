from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02414_0():
    input_content = "3 2 3\n1 2\n0 3\n4 5\n1 2 1\n0 3 2"
    expected_output = "1 8 5\n0 9 6\n4 23 14"
    run_pie_test_case("../p02414.py", input_content, expected_output)


def test_problem_p02414_1():
    input_content = "3 2 3\n1 2\n0 3\n4 5\n1 2 1\n0 3 2"
    expected_output = "1 8 5\n0 9 6\n4 23 14"
    run_pie_test_case("../p02414.py", input_content, expected_output)
