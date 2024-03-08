from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03232_0():
    input_content = "2\n1 2"
    expected_output = "9"
    run_pie_test_case("../p03232.py", input_content, expected_output)


def test_problem_p03232_1():
    input_content = "10\n1 2 4 8 16 32 64 128 256 512"
    expected_output = "880971923"
    run_pie_test_case("../p03232.py", input_content, expected_output)


def test_problem_p03232_2():
    input_content = "2\n1 2"
    expected_output = "9"
    run_pie_test_case("../p03232.py", input_content, expected_output)


def test_problem_p03232_3():
    input_content = "4\n1 1 1 1"
    expected_output = "212"
    run_pie_test_case("../p03232.py", input_content, expected_output)
