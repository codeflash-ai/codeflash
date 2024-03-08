from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00491_0():
    input_content = "5 3\n3 1\n1 1\n4 2"
    expected_output = "6"
    run_pie_test_case("../p00491.py", input_content, expected_output)


def test_problem_p00491_1():
    input_content = "5 3\n3 1\n1 1\n4 2"
    expected_output = "6"
    run_pie_test_case("../p00491.py", input_content, expected_output)
