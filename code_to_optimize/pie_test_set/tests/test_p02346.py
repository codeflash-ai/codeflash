from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02346_0():
    input_content = "3 5\n0 1 1\n0 2 2\n0 3 3\n1 1 2\n1 2 2"
    expected_output = "3\n2"
    run_pie_test_case("../p02346.py", input_content, expected_output)


def test_problem_p02346_1():
    input_content = "3 5\n0 1 1\n0 2 2\n0 3 3\n1 1 2\n1 2 2"
    expected_output = "3\n2"
    run_pie_test_case("../p02346.py", input_content, expected_output)
