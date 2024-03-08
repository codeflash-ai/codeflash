from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03659_0():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "1"
    run_pie_test_case("../p03659.py", input_content, expected_output)


def test_problem_p03659_1():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "1"
    run_pie_test_case("../p03659.py", input_content, expected_output)


def test_problem_p03659_2():
    input_content = "2\n10 -10"
    expected_output = "20"
    run_pie_test_case("../p03659.py", input_content, expected_output)
