from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02837_0():
    input_content = "3\n1\n2 1\n1\n1 1\n1\n2 0"
    expected_output = "2"
    run_pie_test_case("../p02837.py", input_content, expected_output)


def test_problem_p02837_1():
    input_content = "3\n2\n2 1\n3 0\n2\n3 1\n1 0\n2\n1 1\n2 0"
    expected_output = "0"
    run_pie_test_case("../p02837.py", input_content, expected_output)


def test_problem_p02837_2():
    input_content = "3\n1\n2 1\n1\n1 1\n1\n2 0"
    expected_output = "2"
    run_pie_test_case("../p02837.py", input_content, expected_output)


def test_problem_p02837_3():
    input_content = "2\n1\n2 0\n1\n1 0"
    expected_output = "1"
    run_pie_test_case("../p02837.py", input_content, expected_output)
