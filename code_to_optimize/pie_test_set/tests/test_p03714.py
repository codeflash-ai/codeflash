from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03714_0():
    input_content = "2\n3 1 4 1 5 9"
    expected_output = "1"
    run_pie_test_case("../p03714.py", input_content, expected_output)


def test_problem_p03714_1():
    input_content = "2\n3 1 4 1 5 9"
    expected_output = "1"
    run_pie_test_case("../p03714.py", input_content, expected_output)


def test_problem_p03714_2():
    input_content = "1\n1 2 3"
    expected_output = "-1"
    run_pie_test_case("../p03714.py", input_content, expected_output)


def test_problem_p03714_3():
    input_content = "3\n8 2 2 7 4 6 5 3 8"
    expected_output = "5"
    run_pie_test_case("../p03714.py", input_content, expected_output)
