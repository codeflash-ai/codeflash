from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03059_0():
    input_content = "3 5 7"
    expected_output = "10"
    run_pie_test_case("../p03059.py", input_content, expected_output)


def test_problem_p03059_1():
    input_content = "3 5 7"
    expected_output = "10"
    run_pie_test_case("../p03059.py", input_content, expected_output)


def test_problem_p03059_2():
    input_content = "20 20 19"
    expected_output = "0"
    run_pie_test_case("../p03059.py", input_content, expected_output)


def test_problem_p03059_3():
    input_content = "3 2 9"
    expected_output = "6"
    run_pie_test_case("../p03059.py", input_content, expected_output)
