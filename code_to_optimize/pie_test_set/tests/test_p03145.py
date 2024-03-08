from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03145_0():
    input_content = "3 4 5"
    expected_output = "6"
    run_pie_test_case("../p03145.py", input_content, expected_output)


def test_problem_p03145_1():
    input_content = "3 4 5"
    expected_output = "6"
    run_pie_test_case("../p03145.py", input_content, expected_output)


def test_problem_p03145_2():
    input_content = "45 28 53"
    expected_output = "630"
    run_pie_test_case("../p03145.py", input_content, expected_output)


def test_problem_p03145_3():
    input_content = "5 12 13"
    expected_output = "30"
    run_pie_test_case("../p03145.py", input_content, expected_output)
