from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03986_0():
    input_content = "TSTTSS"
    expected_output = "4"
    run_pie_test_case("../p03986.py", input_content, expected_output)


def test_problem_p03986_1():
    input_content = "TSTTSS"
    expected_output = "4"
    run_pie_test_case("../p03986.py", input_content, expected_output)


def test_problem_p03986_2():
    input_content = "SSTTST"
    expected_output = "0"
    run_pie_test_case("../p03986.py", input_content, expected_output)


def test_problem_p03986_3():
    input_content = "TSSTTTSS"
    expected_output = "4"
    run_pie_test_case("../p03986.py", input_content, expected_output)
