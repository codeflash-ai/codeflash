from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03026_0():
    input_content = "5\n1 2\n2 3\n3 4\n4 5\n1 2 3 4 5"
    expected_output = "10\n1 2 3 4 5"
    run_pie_test_case("../p03026.py", input_content, expected_output)


def test_problem_p03026_1():
    input_content = "5\n1 2\n1 3\n1 4\n1 5\n3141 59 26 53 59"
    expected_output = "197\n59 26 3141 59 53"
    run_pie_test_case("../p03026.py", input_content, expected_output)


def test_problem_p03026_2():
    input_content = "5\n1 2\n2 3\n3 4\n4 5\n1 2 3 4 5"
    expected_output = "10\n1 2 3 4 5"
    run_pie_test_case("../p03026.py", input_content, expected_output)
