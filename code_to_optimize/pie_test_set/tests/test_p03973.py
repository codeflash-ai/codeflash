from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03973_0():
    input_content = "3\n3\n2\n5"
    expected_output = "3"
    run_pie_test_case("../p03973.py", input_content, expected_output)


def test_problem_p03973_1():
    input_content = "15\n3\n1\n4\n1\n5\n9\n2\n6\n5\n3\n5\n8\n9\n7\n9"
    expected_output = "18"
    run_pie_test_case("../p03973.py", input_content, expected_output)


def test_problem_p03973_2():
    input_content = "3\n3\n2\n5"
    expected_output = "3"
    run_pie_test_case("../p03973.py", input_content, expected_output)
