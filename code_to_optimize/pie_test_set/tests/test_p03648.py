from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03648_0():
    input_content = "0"
    expected_output = "4\n3 3 3 3"
    run_pie_test_case("../p03648.py", input_content, expected_output)


def test_problem_p03648_1():
    input_content = "0"
    expected_output = "4\n3 3 3 3"
    run_pie_test_case("../p03648.py", input_content, expected_output)


def test_problem_p03648_2():
    input_content = "1"
    expected_output = "3\n1 0 3"
    run_pie_test_case("../p03648.py", input_content, expected_output)


def test_problem_p03648_3():
    input_content = "1234567894848"
    expected_output = "10\n1000 193 256 777 0 1 1192 1234567891011 48 425"
    run_pie_test_case("../p03648.py", input_content, expected_output)


def test_problem_p03648_4():
    input_content = "2"
    expected_output = "2\n2 2"
    run_pie_test_case("../p03648.py", input_content, expected_output)


def test_problem_p03648_5():
    input_content = "3"
    expected_output = "7\n27 0 0 0 0 0 0"
    run_pie_test_case("../p03648.py", input_content, expected_output)
