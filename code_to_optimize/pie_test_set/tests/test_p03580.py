from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03580_0():
    input_content = "7\n1010101"
    expected_output = "2"
    run_pie_test_case("../p03580.py", input_content, expected_output)


def test_problem_p03580_1():
    input_content = "7\n1010101"
    expected_output = "2"
    run_pie_test_case("../p03580.py", input_content, expected_output)


def test_problem_p03580_2():
    input_content = "50\n10101000010011011110001001111110000101010111100110"
    expected_output = "10"
    run_pie_test_case("../p03580.py", input_content, expected_output)
