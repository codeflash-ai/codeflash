from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02300_0():
    input_content = "7\n2 1\n0 0\n1 2\n2 2\n4 2\n1 3\n3 3"
    expected_output = "5\n0 0\n2 1\n4 2\n3 3\n1 3"
    run_pie_test_case("../p02300.py", input_content, expected_output)


def test_problem_p02300_1():
    input_content = "4\n0 0\n2 2\n0 2\n0 1"
    expected_output = "4\n0 0\n2 2\n0 2\n0 1"
    run_pie_test_case("../p02300.py", input_content, expected_output)


def test_problem_p02300_2():
    input_content = "7\n2 1\n0 0\n1 2\n2 2\n4 2\n1 3\n3 3"
    expected_output = "5\n0 0\n2 1\n4 2\n3 3\n1 3"
    run_pie_test_case("../p02300.py", input_content, expected_output)
