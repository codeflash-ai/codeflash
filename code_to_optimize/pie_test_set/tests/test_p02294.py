from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02294_0():
    input_content = "3\n0 0 3 0 1 1 2 -1\n0 0 3 0 3 1 3 -1\n0 0 3 0 3 -2 5 0"
    expected_output = "1\n1\n0"
    run_pie_test_case("../p02294.py", input_content, expected_output)


def test_problem_p02294_1():
    input_content = "3\n0 0 3 0 1 1 2 -1\n0 0 3 0 3 1 3 -1\n0 0 3 0 3 -2 5 0"
    expected_output = "1\n1\n0"
    run_pie_test_case("../p02294.py", input_content, expected_output)
