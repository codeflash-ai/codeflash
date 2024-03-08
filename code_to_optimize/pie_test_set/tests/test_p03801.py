from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03801_0():
    input_content = "2\n1 2"
    expected_output = "2\n1"
    run_pie_test_case("../p03801.py", input_content, expected_output)


def test_problem_p03801_1():
    input_content = "2\n1 2"
    expected_output = "2\n1"
    run_pie_test_case("../p03801.py", input_content, expected_output)


def test_problem_p03801_2():
    input_content = "10\n1 2 1 3 2 4 2 5 8 1"
    expected_output = "10\n7\n0\n4\n0\n3\n0\n2\n3\n0"
    run_pie_test_case("../p03801.py", input_content, expected_output)
