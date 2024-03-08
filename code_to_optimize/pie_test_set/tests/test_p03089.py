from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03089_0():
    input_content = "3\n1 2 1"
    expected_output = "1\n1\n2"
    run_pie_test_case("../p03089.py", input_content, expected_output)


def test_problem_p03089_1():
    input_content = "9\n1 1 1 2 2 1 2 3 2"
    expected_output = "1\n2\n2\n3\n1\n2\n2\n1\n1"
    run_pie_test_case("../p03089.py", input_content, expected_output)


def test_problem_p03089_2():
    input_content = "2\n2 2"
    expected_output = "-1"
    run_pie_test_case("../p03089.py", input_content, expected_output)


def test_problem_p03089_3():
    input_content = "3\n1 2 1"
    expected_output = "1\n1\n2"
    run_pie_test_case("../p03089.py", input_content, expected_output)
