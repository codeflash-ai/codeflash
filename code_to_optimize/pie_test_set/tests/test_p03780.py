from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03780_0():
    input_content = "3 6\n1 4 3"
    expected_output = "1"
    run_pie_test_case("../p03780.py", input_content, expected_output)


def test_problem_p03780_1():
    input_content = "5 400\n3 1 4 1 5"
    expected_output = "5"
    run_pie_test_case("../p03780.py", input_content, expected_output)


def test_problem_p03780_2():
    input_content = "3 6\n1 4 3"
    expected_output = "1"
    run_pie_test_case("../p03780.py", input_content, expected_output)


def test_problem_p03780_3():
    input_content = "6 20\n10 4 3 10 25 2"
    expected_output = "3"
    run_pie_test_case("../p03780.py", input_content, expected_output)
