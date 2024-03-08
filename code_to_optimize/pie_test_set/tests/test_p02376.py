from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02376_0():
    input_content = "4 5\n0 1 2\n0 2 1\n1 2 1\n1 3 1\n2 3 2"
    expected_output = "3"
    run_pie_test_case("../p02376.py", input_content, expected_output)


def test_problem_p02376_1():
    input_content = "4 5\n0 1 2\n0 2 1\n1 2 1\n1 3 1\n2 3 2"
    expected_output = "3"
    run_pie_test_case("../p02376.py", input_content, expected_output)
