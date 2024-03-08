from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03092_0():
    input_content = "3 20 30\n3 1 2"
    expected_output = "20"
    run_pie_test_case("../p03092.py", input_content, expected_output)


def test_problem_p03092_1():
    input_content = "3 20 30\n3 1 2"
    expected_output = "20"
    run_pie_test_case("../p03092.py", input_content, expected_output)


def test_problem_p03092_2():
    input_content = "1 10 10\n1"
    expected_output = "0"
    run_pie_test_case("../p03092.py", input_content, expected_output)


def test_problem_p03092_3():
    input_content = "4 20 30\n4 2 3 1"
    expected_output = "50"
    run_pie_test_case("../p03092.py", input_content, expected_output)


def test_problem_p03092_4():
    input_content = "4 1000000000 1000000000\n4 3 2 1"
    expected_output = "3000000000"
    run_pie_test_case("../p03092.py", input_content, expected_output)


def test_problem_p03092_5():
    input_content = "9 40 50\n5 3 4 7 6 1 2 9 8"
    expected_output = "220"
    run_pie_test_case("../p03092.py", input_content, expected_output)
