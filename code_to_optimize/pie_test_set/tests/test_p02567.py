from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02567_0():
    input_content = "5 5\n1 2 3 2 1\n2 1 5\n3 2 3\n1 3 1\n2 2 4\n3 1 3"
    expected_output = "3\n3\n2\n6"
    run_pie_test_case("../p02567.py", input_content, expected_output)


def test_problem_p02567_1():
    input_content = "5 5\n1 2 3 2 1\n2 1 5\n3 2 3\n1 3 1\n2 2 4\n3 1 3"
    expected_output = "3\n3\n2\n6"
    run_pie_test_case("../p02567.py", input_content, expected_output)
