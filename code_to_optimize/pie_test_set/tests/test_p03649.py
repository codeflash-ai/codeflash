from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03649_0():
    input_content = "4\n3 3 3 3"
    expected_output = "0"
    run_pie_test_case("../p03649.py", input_content, expected_output)


def test_problem_p03649_1():
    input_content = "2\n2 2"
    expected_output = "2"
    run_pie_test_case("../p03649.py", input_content, expected_output)


def test_problem_p03649_2():
    input_content = "4\n3 3 3 3"
    expected_output = "0"
    run_pie_test_case("../p03649.py", input_content, expected_output)


def test_problem_p03649_3():
    input_content = "7\n27 0 0 0 0 0 0"
    expected_output = "3"
    run_pie_test_case("../p03649.py", input_content, expected_output)


def test_problem_p03649_4():
    input_content = "3\n1 0 3"
    expected_output = "1"
    run_pie_test_case("../p03649.py", input_content, expected_output)


def test_problem_p03649_5():
    input_content = "10\n1000 193 256 777 0 1 1192 1234567891011 48 425"
    expected_output = "1234567894848"
    run_pie_test_case("../p03649.py", input_content, expected_output)
