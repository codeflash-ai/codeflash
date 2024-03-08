from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03228_0():
    input_content = "5 4 2"
    expected_output = "5 3"
    run_pie_test_case("../p03228.py", input_content, expected_output)


def test_problem_p03228_1():
    input_content = "3 3 3"
    expected_output = "1 3"
    run_pie_test_case("../p03228.py", input_content, expected_output)


def test_problem_p03228_2():
    input_content = "5 4 2"
    expected_output = "5 3"
    run_pie_test_case("../p03228.py", input_content, expected_output)


def test_problem_p03228_3():
    input_content = "314159265 358979323 84"
    expected_output = "448759046 224379523"
    run_pie_test_case("../p03228.py", input_content, expected_output)
