from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02912_0():
    input_content = "3 3\n2 13 8"
    expected_output = "9"
    run_pie_test_case("../p02912.py", input_content, expected_output)


def test_problem_p02912_1():
    input_content = "1 100000\n1000000000"
    expected_output = "0"
    run_pie_test_case("../p02912.py", input_content, expected_output)


def test_problem_p02912_2():
    input_content = "10 1\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "9500000000"
    run_pie_test_case("../p02912.py", input_content, expected_output)


def test_problem_p02912_3():
    input_content = "4 4\n1 9 3 5"
    expected_output = "6"
    run_pie_test_case("../p02912.py", input_content, expected_output)


def test_problem_p02912_4():
    input_content = "3 3\n2 13 8"
    expected_output = "9"
    run_pie_test_case("../p02912.py", input_content, expected_output)
