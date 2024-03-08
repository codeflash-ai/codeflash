from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03420_0():
    input_content = "5 2"
    expected_output = "7"
    run_pie_test_case("../p03420.py", input_content, expected_output)


def test_problem_p03420_1():
    input_content = "5 2"
    expected_output = "7"
    run_pie_test_case("../p03420.py", input_content, expected_output)


def test_problem_p03420_2():
    input_content = "31415 9265"
    expected_output = "287927211"
    run_pie_test_case("../p03420.py", input_content, expected_output)


def test_problem_p03420_3():
    input_content = "10 0"
    expected_output = "100"
    run_pie_test_case("../p03420.py", input_content, expected_output)
