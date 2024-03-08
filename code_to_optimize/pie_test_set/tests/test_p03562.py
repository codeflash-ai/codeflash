from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03562_0():
    input_content = "3 111\n1111\n10111\n10010"
    expected_output = "4"
    run_pie_test_case("../p03562.py", input_content, expected_output)


def test_problem_p03562_1():
    input_content = "1 111111111111111111111111111111111111111111111111111111111111111\n1"
    expected_output = "466025955"
    run_pie_test_case("../p03562.py", input_content, expected_output)


def test_problem_p03562_2():
    input_content = "4 100100\n1011\n1110\n110101\n1010110"
    expected_output = "37"
    run_pie_test_case("../p03562.py", input_content, expected_output)


def test_problem_p03562_3():
    input_content = "4 111001100101001\n10111110\n1001000110\n100000101\n11110000011"
    expected_output = "1843"
    run_pie_test_case("../p03562.py", input_content, expected_output)


def test_problem_p03562_4():
    input_content = "3 111\n1111\n10111\n10010"
    expected_output = "4"
    run_pie_test_case("../p03562.py", input_content, expected_output)
