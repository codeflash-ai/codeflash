from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02635_0():
    input_content = "0101 1"
    expected_output = "4"
    run_pie_test_case("../p02635.py", input_content, expected_output)


def test_problem_p02635_1():
    input_content = "0101 1"
    expected_output = "4"
    run_pie_test_case("../p02635.py", input_content, expected_output)


def test_problem_p02635_2():
    input_content = "01100110 2"
    expected_output = "14"
    run_pie_test_case("../p02635.py", input_content, expected_output)


def test_problem_p02635_3():
    input_content = "1101010010101101110111100011011111011000111101110101010010101010101 20"
    expected_output = "113434815"
    run_pie_test_case("../p02635.py", input_content, expected_output)
