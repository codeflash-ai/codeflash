from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03953_0():
    input_content = "3\n-1 0 2\n1 1\n2"
    expected_output = "-1.0\n1.0\n2.0"
    run_pie_test_case("../p03953.py", input_content, expected_output)


def test_problem_p03953_1():
    input_content = "3\n1 -1 1\n2 2\n2 2"
    expected_output = "1.0\n-1.0\n1.0"
    run_pie_test_case("../p03953.py", input_content, expected_output)


def test_problem_p03953_2():
    input_content = "3\n-1 0 2\n1 1\n2"
    expected_output = "-1.0\n1.0\n2.0"
    run_pie_test_case("../p03953.py", input_content, expected_output)


def test_problem_p03953_3():
    input_content = "5\n0 1 3 6 10\n3 10\n2 3 4"
    expected_output = "0.0\n3.0\n7.0\n8.0\n10.0"
    run_pie_test_case("../p03953.py", input_content, expected_output)
