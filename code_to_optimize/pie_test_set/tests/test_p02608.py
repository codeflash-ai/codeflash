from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02608_0():
    input_content = "20"
    expected_output = "0\n0\n0\n0\n0\n1\n0\n0\n0\n0\n3\n0\n0\n0\n0\n0\n3\n3\n0\n0"
    run_pie_test_case("../p02608.py", input_content, expected_output)


def test_problem_p02608_1():
    input_content = "20"
    expected_output = "0\n0\n0\n0\n0\n1\n0\n0\n0\n0\n3\n0\n0\n0\n0\n0\n3\n3\n0\n0"
    run_pie_test_case("../p02608.py", input_content, expected_output)
