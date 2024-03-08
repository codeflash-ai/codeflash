from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03806_0():
    input_content = "3 1 1\n1 2 1\n2 1 2\n3 3 10"
    expected_output = "3"
    run_pie_test_case("../p03806.py", input_content, expected_output)


def test_problem_p03806_1():
    input_content = "1 1 10\n10 10 10"
    expected_output = "-1"
    run_pie_test_case("../p03806.py", input_content, expected_output)


def test_problem_p03806_2():
    input_content = "3 1 1\n1 2 1\n2 1 2\n3 3 10"
    expected_output = "3"
    run_pie_test_case("../p03806.py", input_content, expected_output)
