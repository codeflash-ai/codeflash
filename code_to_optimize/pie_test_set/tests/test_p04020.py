from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04020_0():
    input_content = "4\n4\n0\n3\n2"
    expected_output = "4"
    run_pie_test_case("../p04020.py", input_content, expected_output)


def test_problem_p04020_1():
    input_content = "8\n2\n0\n1\n6\n0\n8\n2\n1"
    expected_output = "9"
    run_pie_test_case("../p04020.py", input_content, expected_output)


def test_problem_p04020_2():
    input_content = "4\n4\n0\n3\n2"
    expected_output = "4"
    run_pie_test_case("../p04020.py", input_content, expected_output)
