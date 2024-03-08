from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03597_0():
    input_content = "3\n4"
    expected_output = "5"
    run_pie_test_case("../p03597.py", input_content, expected_output)


def test_problem_p03597_1():
    input_content = "3\n4"
    expected_output = "5"
    run_pie_test_case("../p03597.py", input_content, expected_output)


def test_problem_p03597_2():
    input_content = "19\n100"
    expected_output = "261"
    run_pie_test_case("../p03597.py", input_content, expected_output)


def test_problem_p03597_3():
    input_content = "10\n0"
    expected_output = "100"
    run_pie_test_case("../p03597.py", input_content, expected_output)
