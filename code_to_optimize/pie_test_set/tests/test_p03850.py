from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03850_0():
    input_content = "3\n5 - 1 - 3"
    expected_output = "7"
    run_pie_test_case("../p03850.py", input_content, expected_output)


def test_problem_p03850_1():
    input_content = "5\n1 - 2 + 3 - 4 + 5"
    expected_output = "5"
    run_pie_test_case("../p03850.py", input_content, expected_output)


def test_problem_p03850_2():
    input_content = "3\n5 - 1 - 3"
    expected_output = "7"
    run_pie_test_case("../p03850.py", input_content, expected_output)


def test_problem_p03850_3():
    input_content = "5\n1 - 20 - 13 + 14 - 5"
    expected_output = "13"
    run_pie_test_case("../p03850.py", input_content, expected_output)
