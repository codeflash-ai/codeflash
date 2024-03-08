from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03791_0():
    input_content = "3\n1 2 3"
    expected_output = "4"
    run_pie_test_case("../p03791.py", input_content, expected_output)


def test_problem_p03791_1():
    input_content = "13\n4 6 8 9 10 12 14 15 16 18 20 21 22"
    expected_output = "311014372"
    run_pie_test_case("../p03791.py", input_content, expected_output)


def test_problem_p03791_2():
    input_content = "3\n1 2 3"
    expected_output = "4"
    run_pie_test_case("../p03791.py", input_content, expected_output)


def test_problem_p03791_3():
    input_content = "3\n2 3 4"
    expected_output = "6"
    run_pie_test_case("../p03791.py", input_content, expected_output)


def test_problem_p03791_4():
    input_content = "8\n1 2 3 5 7 11 13 17"
    expected_output = "10080"
    run_pie_test_case("../p03791.py", input_content, expected_output)
