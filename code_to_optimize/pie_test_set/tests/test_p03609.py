from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03609_0():
    input_content = "100 17"
    expected_output = "83"
    run_pie_test_case("../p03609.py", input_content, expected_output)


def test_problem_p03609_1():
    input_content = "48 58"
    expected_output = "0"
    run_pie_test_case("../p03609.py", input_content, expected_output)


def test_problem_p03609_2():
    input_content = "1000000000 1000000000"
    expected_output = "0"
    run_pie_test_case("../p03609.py", input_content, expected_output)


def test_problem_p03609_3():
    input_content = "100 17"
    expected_output = "83"
    run_pie_test_case("../p03609.py", input_content, expected_output)
