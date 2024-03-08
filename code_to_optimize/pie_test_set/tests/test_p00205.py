from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00205_0():
    input_content = "1\n2\n3\n2\n1\n1\n2\n2\n2\n1\n0"
    expected_output = "3\n3\n3\n3\n3\n1\n2\n2\n2\n1"
    run_pie_test_case("../p00205.py", input_content, expected_output)


def test_problem_p00205_1():
    input_content = "1\n2\n3\n2\n1\n1\n2\n2\n2\n1\n0"
    expected_output = "3\n3\n3\n3\n3\n1\n2\n2\n2\n1"
    run_pie_test_case("../p00205.py", input_content, expected_output)
