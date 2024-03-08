from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02431_0():
    input_content = "8\n0 1\n0 2\n0 3\n2\n0 4\n1 0\n1 1\n1 2"
    expected_output = "1\n2\n4"
    run_pie_test_case("../p02431.py", input_content, expected_output)


def test_problem_p02431_1():
    input_content = "8\n0 1\n0 2\n0 3\n2\n0 4\n1 0\n1 1\n1 2"
    expected_output = "1\n2\n4"
    run_pie_test_case("../p02431.py", input_content, expected_output)
