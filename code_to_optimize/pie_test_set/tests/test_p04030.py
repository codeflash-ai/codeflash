from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04030_0():
    input_content = "01B0"
    expected_output = "00"
    run_pie_test_case("../p04030.py", input_content, expected_output)


def test_problem_p04030_1():
    input_content = "01B0"
    expected_output = "00"
    run_pie_test_case("../p04030.py", input_content, expected_output)


def test_problem_p04030_2():
    input_content = "0BB1"
    expected_output = "1"
    run_pie_test_case("../p04030.py", input_content, expected_output)
