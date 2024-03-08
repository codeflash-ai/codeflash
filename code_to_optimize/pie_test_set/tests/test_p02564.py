from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02564_0():
    input_content = "6 7\n1 4\n5 2\n3 0\n5 5\n4 1\n0 3\n4 2"
    expected_output = "4\n1 5\n2 4 1\n1 2\n2 3 0"
    run_pie_test_case("../p02564.py", input_content, expected_output)


def test_problem_p02564_1():
    input_content = "6 7\n1 4\n5 2\n3 0\n5 5\n4 1\n0 3\n4 2"
    expected_output = "4\n1 5\n2 4 1\n1 2\n2 3 0"
    run_pie_test_case("../p02564.py", input_content, expected_output)
