from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03343_0():
    input_content = "5 3 2\n4 3 1 5 2"
    expected_output = "1"
    run_pie_test_case("../p03343.py", input_content, expected_output)


def test_problem_p03343_1():
    input_content = "11 7 5\n24979445 861648772 623690081 433933447 476190629 262703497 211047202 971407775 628894325 731963982 822804784"
    expected_output = "451211184"
    run_pie_test_case("../p03343.py", input_content, expected_output)


def test_problem_p03343_2():
    input_content = "5 3 2\n4 3 1 5 2"
    expected_output = "1"
    run_pie_test_case("../p03343.py", input_content, expected_output)


def test_problem_p03343_3():
    input_content = "10 1 6\n1 1 2 3 5 8 13 21 34 55"
    expected_output = "7"
    run_pie_test_case("../p03343.py", input_content, expected_output)
