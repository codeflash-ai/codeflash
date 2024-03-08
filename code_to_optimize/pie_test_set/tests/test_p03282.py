from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03282_0():
    input_content = "1214\n4"
    expected_output = "2"
    run_pie_test_case("../p03282.py", input_content, expected_output)


def test_problem_p03282_1():
    input_content = "299792458\n9460730472580800"
    expected_output = "2"
    run_pie_test_case("../p03282.py", input_content, expected_output)


def test_problem_p03282_2():
    input_content = "3\n157"
    expected_output = "3"
    run_pie_test_case("../p03282.py", input_content, expected_output)


def test_problem_p03282_3():
    input_content = "1214\n4"
    expected_output = "2"
    run_pie_test_case("../p03282.py", input_content, expected_output)
