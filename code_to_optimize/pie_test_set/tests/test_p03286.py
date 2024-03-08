from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03286_0():
    input_content = "-9"
    expected_output = "1011"
    run_pie_test_case("../p03286.py", input_content, expected_output)


def test_problem_p03286_1():
    input_content = "0"
    expected_output = "0"
    run_pie_test_case("../p03286.py", input_content, expected_output)


def test_problem_p03286_2():
    input_content = "123456789"
    expected_output = "11000101011001101110100010101"
    run_pie_test_case("../p03286.py", input_content, expected_output)


def test_problem_p03286_3():
    input_content = "-9"
    expected_output = "1011"
    run_pie_test_case("../p03286.py", input_content, expected_output)
