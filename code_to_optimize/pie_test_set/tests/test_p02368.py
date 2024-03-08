from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02368_0():
    input_content = "5 6\n0 1\n1 0\n1 2\n2 4\n4 3\n3 2\n4\n0 1\n0 3\n2 3\n3 4"
    expected_output = "1\n0\n1\n1"
    run_pie_test_case("../p02368.py", input_content, expected_output)


def test_problem_p02368_1():
    input_content = "5 6\n0 1\n1 0\n1 2\n2 4\n4 3\n3 2\n4\n0 1\n0 3\n2 3\n3 4"
    expected_output = "1\n0\n1\n1"
    run_pie_test_case("../p02368.py", input_content, expected_output)
