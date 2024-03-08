from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02432_0():
    input_content = "11\n0 0 1\n0 0 2\n0 1 3\n1 0\n1 1\n1 2\n2 0\n2 1\n0 0 4\n1 0\n1 1"
    expected_output = "2\n1\n3\n4\n1"
    run_pie_test_case("../p02432.py", input_content, expected_output)


def test_problem_p02432_1():
    input_content = "11\n0 0 1\n0 0 2\n0 1 3\n1 0\n1 1\n1 2\n2 0\n2 1\n0 0 4\n1 0\n1 1"
    expected_output = "2\n1\n3\n4\n1"
    run_pie_test_case("../p02432.py", input_content, expected_output)
