from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03489_0():
    input_content = "4\n3 3 3 3"
    expected_output = "1"
    run_pie_test_case("../p03489.py", input_content, expected_output)


def test_problem_p03489_1():
    input_content = "8\n2 7 1 8 2 8 1 8"
    expected_output = "5"
    run_pie_test_case("../p03489.py", input_content, expected_output)


def test_problem_p03489_2():
    input_content = "1\n1000000000"
    expected_output = "1"
    run_pie_test_case("../p03489.py", input_content, expected_output)


def test_problem_p03489_3():
    input_content = "6\n1 2 2 3 3 3"
    expected_output = "0"
    run_pie_test_case("../p03489.py", input_content, expected_output)


def test_problem_p03489_4():
    input_content = "4\n3 3 3 3"
    expected_output = "1"
    run_pie_test_case("../p03489.py", input_content, expected_output)


def test_problem_p03489_5():
    input_content = "5\n2 4 1 4 2"
    expected_output = "2"
    run_pie_test_case("../p03489.py", input_content, expected_output)
