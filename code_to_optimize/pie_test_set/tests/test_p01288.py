from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01288_0():
    input_content = "6 3\n1\n1\n2\n3\n3\nQ 5\nM 3\nQ 5\n0 0"
    expected_output = "4"
    run_pie_test_case("../p01288.py", input_content, expected_output)


def test_problem_p01288_1():
    input_content = "6 3\n1\n1\n2\n3\n3\nQ 5\nM 3\nQ 5\n0 0"
    expected_output = "4"
    run_pie_test_case("../p01288.py", input_content, expected_output)
