from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03416_0():
    input_content = "11009 11332"
    expected_output = "4"
    run_pie_test_case("../p03416.py", input_content, expected_output)


def test_problem_p03416_1():
    input_content = "11009 11332"
    expected_output = "4"
    run_pie_test_case("../p03416.py", input_content, expected_output)


def test_problem_p03416_2():
    input_content = "31415 92653"
    expected_output = "612"
    run_pie_test_case("../p03416.py", input_content, expected_output)
