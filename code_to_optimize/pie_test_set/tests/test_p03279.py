from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03279_0():
    input_content = "2 2\n2 3\n1 4"
    expected_output = "3"
    run_pie_test_case("../p03279.py", input_content, expected_output)


def test_problem_p03279_1():
    input_content = "4 1\n1 2 4 5\n3"
    expected_output = "1"
    run_pie_test_case("../p03279.py", input_content, expected_output)


def test_problem_p03279_2():
    input_content = "4 5\n2 5 7 11\n1 3 6 9 13"
    expected_output = "6"
    run_pie_test_case("../p03279.py", input_content, expected_output)


def test_problem_p03279_3():
    input_content = "10 10\n4 13 15 18 19 20 21 22 25 27\n1 5 11 12 14 16 23 26 29 30"
    expected_output = "22"
    run_pie_test_case("../p03279.py", input_content, expected_output)


def test_problem_p03279_4():
    input_content = "3 4\n2 5 10\n1 3 7 13"
    expected_output = "8"
    run_pie_test_case("../p03279.py", input_content, expected_output)


def test_problem_p03279_5():
    input_content = "2 2\n2 3\n1 4"
    expected_output = "3"
    run_pie_test_case("../p03279.py", input_content, expected_output)
