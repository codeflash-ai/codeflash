from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03880_0():
    input_content = "3\n2\n3\n4"
    expected_output = "3"
    run_pie_test_case("../p03880.py", input_content, expected_output)


def test_problem_p03880_1():
    input_content = "3\n2\n3\n4"
    expected_output = "3"
    run_pie_test_case("../p03880.py", input_content, expected_output)


def test_problem_p03880_2():
    input_content = "3\n100\n100\n100"
    expected_output = "-1"
    run_pie_test_case("../p03880.py", input_content, expected_output)
