from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02299_0():
    input_content = "4\n0 0\n3 1\n2 3\n0 3\n3\n2 1\n0 2\n3 2"
    expected_output = "2\n1\n0"
    run_pie_test_case("../p02299.py", input_content, expected_output)


def test_problem_p02299_1():
    input_content = "4\n0 0\n3 1\n2 3\n0 3\n3\n2 1\n0 2\n3 2"
    expected_output = "2\n1\n0"
    run_pie_test_case("../p02299.py", input_content, expected_output)
