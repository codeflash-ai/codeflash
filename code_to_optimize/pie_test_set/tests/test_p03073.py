from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03073_0():
    input_content = "000"
    expected_output = "1"
    run_pie_test_case("../p03073.py", input_content, expected_output)


def test_problem_p03073_1():
    input_content = "10010010"
    expected_output = "3"
    run_pie_test_case("../p03073.py", input_content, expected_output)


def test_problem_p03073_2():
    input_content = "0"
    expected_output = "0"
    run_pie_test_case("../p03073.py", input_content, expected_output)


def test_problem_p03073_3():
    input_content = "000"
    expected_output = "1"
    run_pie_test_case("../p03073.py", input_content, expected_output)
