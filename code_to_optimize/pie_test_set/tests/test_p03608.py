from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03608_0():
    input_content = "3 3 3\n1 2 3\n1 2 1\n2 3 1\n3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03608.py", input_content, expected_output)


def test_problem_p03608_1():
    input_content = "4 6 3\n2 3 4\n1 2 4\n2 3 3\n4 3 1\n1 4 1\n4 2 2\n3 1 6"
    expected_output = "3"
    run_pie_test_case("../p03608.py", input_content, expected_output)


def test_problem_p03608_2():
    input_content = "3 3 3\n1 2 3\n1 2 1\n2 3 1\n3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03608.py", input_content, expected_output)


def test_problem_p03608_3():
    input_content = "3 3 2\n1 3\n2 3 2\n1 3 6\n1 2 2"
    expected_output = "4"
    run_pie_test_case("../p03608.py", input_content, expected_output)
