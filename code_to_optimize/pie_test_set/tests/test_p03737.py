from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03737_0():
    input_content = "atcoder beginner contest"
    expected_output = "ABC"
    run_pie_test_case("../p03737.py", input_content, expected_output)


def test_problem_p03737_1():
    input_content = "resident register number"
    expected_output = "RRN"
    run_pie_test_case("../p03737.py", input_content, expected_output)


def test_problem_p03737_2():
    input_content = "async layered coding"
    expected_output = "ALC"
    run_pie_test_case("../p03737.py", input_content, expected_output)


def test_problem_p03737_3():
    input_content = "k nearest neighbor"
    expected_output = "KNN"
    run_pie_test_case("../p03737.py", input_content, expected_output)


def test_problem_p03737_4():
    input_content = "atcoder beginner contest"
    expected_output = "ABC"
    run_pie_test_case("../p03737.py", input_content, expected_output)
