from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02558_0():
    input_content = "4 7\n1 0 1\n0 0 1\n0 2 3\n1 0 1\n1 1 2\n0 0 2\n1 1 3"
    expected_output = "0\n1\n0\n1"
    run_pie_test_case("../p02558.py", input_content, expected_output)


def test_problem_p02558_1():
    input_content = "4 7\n1 0 1\n0 0 1\n0 2 3\n1 0 1\n1 1 2\n0 0 2\n1 1 3"
    expected_output = "0\n1\n0\n1"
    run_pie_test_case("../p02558.py", input_content, expected_output)
