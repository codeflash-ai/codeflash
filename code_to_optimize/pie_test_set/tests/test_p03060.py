from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03060_0():
    input_content = "3\n10 2 5\n6 3 4"
    expected_output = "5"
    run_pie_test_case("../p03060.py", input_content, expected_output)


def test_problem_p03060_1():
    input_content = "4\n13 21 6 19\n11 30 6 15"
    expected_output = "6"
    run_pie_test_case("../p03060.py", input_content, expected_output)


def test_problem_p03060_2():
    input_content = "3\n10 2 5\n6 3 4"
    expected_output = "5"
    run_pie_test_case("../p03060.py", input_content, expected_output)


def test_problem_p03060_3():
    input_content = "1\n1\n50"
    expected_output = "0"
    run_pie_test_case("../p03060.py", input_content, expected_output)
