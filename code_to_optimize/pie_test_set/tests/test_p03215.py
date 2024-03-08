from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03215_0():
    input_content = "4 2\n2 5 2 5"
    expected_output = "12"
    run_pie_test_case("../p03215.py", input_content, expected_output)


def test_problem_p03215_1():
    input_content = "4 2\n2 5 2 5"
    expected_output = "12"
    run_pie_test_case("../p03215.py", input_content, expected_output)


def test_problem_p03215_2():
    input_content = "8 4\n9 1 8 2 7 5 6 4"
    expected_output = "32"
    run_pie_test_case("../p03215.py", input_content, expected_output)
