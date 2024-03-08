from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03814_0():
    input_content = "QWERTYASDFZXCV"
    expected_output = "5"
    run_pie_test_case("../p03814.py", input_content, expected_output)


def test_problem_p03814_1():
    input_content = "HASFJGHOGAKZZFEGA"
    expected_output = "12"
    run_pie_test_case("../p03814.py", input_content, expected_output)


def test_problem_p03814_2():
    input_content = "ZABCZ"
    expected_output = "4"
    run_pie_test_case("../p03814.py", input_content, expected_output)


def test_problem_p03814_3():
    input_content = "QWERTYASDFZXCV"
    expected_output = "5"
    run_pie_test_case("../p03814.py", input_content, expected_output)
