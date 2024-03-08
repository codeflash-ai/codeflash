from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02575_0():
    input_content = "4 4\n2 4\n1 1\n2 3\n2 4"
    expected_output = "1\n3\n6\n-1"
    run_pie_test_case("../p02575.py", input_content, expected_output)


def test_problem_p02575_1():
    input_content = "4 4\n2 4\n1 1\n2 3\n2 4"
    expected_output = "1\n3\n6\n-1"
    run_pie_test_case("../p02575.py", input_content, expected_output)
