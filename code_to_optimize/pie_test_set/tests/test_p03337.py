from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03337_0():
    input_content = "3 1"
    expected_output = "4"
    run_pie_test_case("../p03337.py", input_content, expected_output)


def test_problem_p03337_1():
    input_content = "0 0"
    expected_output = "0"
    run_pie_test_case("../p03337.py", input_content, expected_output)


def test_problem_p03337_2():
    input_content = "3 1"
    expected_output = "4"
    run_pie_test_case("../p03337.py", input_content, expected_output)


def test_problem_p03337_3():
    input_content = "4 -2"
    expected_output = "6"
    run_pie_test_case("../p03337.py", input_content, expected_output)
