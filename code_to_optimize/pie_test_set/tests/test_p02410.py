from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02410_0():
    input_content = "3 4\n1 2 0 1\n0 3 0 1\n4 1 1 0\n1\n2\n3\n0"
    expected_output = "5\n6\n9"
    run_pie_test_case("../p02410.py", input_content, expected_output)


def test_problem_p02410_1():
    input_content = "3 4\n1 2 0 1\n0 3 0 1\n4 1 1 0\n1\n2\n3\n0"
    expected_output = "5\n6\n9"
    run_pie_test_case("../p02410.py", input_content, expected_output)
