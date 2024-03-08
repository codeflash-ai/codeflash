from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00082_0():
    input_content = "2 3 1 4 0 1 0 1\n4 2 3 2 2 2 1 1"
    expected_output = "1 4 1 4 1 2 1 2\n4 1 4 1 2 1 2 1"
    run_pie_test_case("../p00082.py", input_content, expected_output)


def test_problem_p00082_1():
    input_content = "2 3 1 4 0 1 0 1\n4 2 3 2 2 2 1 1"
    expected_output = "1 4 1 4 1 2 1 2\n4 1 4 1 2 1 2 1"
    run_pie_test_case("../p00082.py", input_content, expected_output)
