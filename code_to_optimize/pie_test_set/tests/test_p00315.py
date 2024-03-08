from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00315_0():
    input_content = "7 8\n00100000\n00011000\n10111101\n01100110\n01000110\n10111101\n00011000\n00100100\n2\n5 3\n1 6\n1\n6 8\n3\n6 8\n3 3\n3 6\n2\n6 3\n6 6\n0\n2\n3 8\n6 8"
    expected_output = "3"
    run_pie_test_case("../p00315.py", input_content, expected_output)


def test_problem_p00315_1():
    input_content = "1 6\n000000\n000000\n010010\n010010\n000000\n000000"
    expected_output = "1"
    run_pie_test_case("../p00315.py", input_content, expected_output)


def test_problem_p00315_2():
    input_content = "7 8\n00100000\n00011000\n10111101\n01100110\n01000110\n10111101\n00011000\n00100100\n2\n5 3\n1 6\n1\n6 8\n3\n6 8\n3 3\n3 6\n2\n6 3\n6 6\n0\n2\n3 8\n6 8"
    expected_output = "3"
    run_pie_test_case("../p00315.py", input_content, expected_output)


def test_problem_p00315_3():
    input_content = "2 2\n00\n00\n4\n1 1\n1 2\n2 1\n2 2"
    expected_output = "2"
    run_pie_test_case("../p00315.py", input_content, expected_output)
