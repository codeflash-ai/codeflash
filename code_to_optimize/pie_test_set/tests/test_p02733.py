from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02733_0():
    input_content = "3 5 4\n11100\n10001\n00111"
    expected_output = "2"
    run_pie_test_case("../p02733.py", input_content, expected_output)


def test_problem_p02733_1():
    input_content = "4 10 4\n1110010010\n1000101110\n0011101001\n1101000111"
    expected_output = "3"
    run_pie_test_case("../p02733.py", input_content, expected_output)


def test_problem_p02733_2():
    input_content = "3 5 8\n11100\n10001\n00111"
    expected_output = "0"
    run_pie_test_case("../p02733.py", input_content, expected_output)


def test_problem_p02733_3():
    input_content = "3 5 4\n11100\n10001\n00111"
    expected_output = "2"
    run_pie_test_case("../p02733.py", input_content, expected_output)
