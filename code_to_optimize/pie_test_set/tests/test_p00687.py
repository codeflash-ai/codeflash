from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00687_0():
    input_content = "10 2 3\n10 2 5\n100 5 25\n0 0 0"
    expected_output = "1\n2\n80"
    run_pie_test_case("../p00687.py", input_content, expected_output)


def test_problem_p00687_1():
    input_content = "10 2 3\n10 2 5\n100 5 25\n0 0 0"
    expected_output = "1\n2\n80"
    run_pie_test_case("../p00687.py", input_content, expected_output)
