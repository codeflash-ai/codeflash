from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02239_0():
    input_content = "4\n1 2 2 4\n2 1 4\n3 0\n4 1 3"
    expected_output = "1 0\n2 1\n3 2\n4 1"
    run_pie_test_case("../p02239.py", input_content, expected_output)


def test_problem_p02239_1():
    input_content = "4\n1 2 2 4\n2 1 4\n3 0\n4 1 3"
    expected_output = "1 0\n2 1\n3 2\n4 1"
    run_pie_test_case("../p02239.py", input_content, expected_output)
