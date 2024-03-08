from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00790_0():
    input_content = "1\nnorth\n3\nnorth\neast\nsouth\n0"
    expected_output = "5\n1"
    run_pie_test_case("../p00790.py", input_content, expected_output)


def test_problem_p00790_1():
    input_content = "1\nnorth\n3\nnorth\neast\nsouth\n0"
    expected_output = "5\n1"
    run_pie_test_case("../p00790.py", input_content, expected_output)
