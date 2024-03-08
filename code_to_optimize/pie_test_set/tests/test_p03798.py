from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03798_0():
    input_content = "6\nooxoox"
    expected_output = "SSSWWS"
    run_pie_test_case("../p03798.py", input_content, expected_output)


def test_problem_p03798_1():
    input_content = "10\noxooxoxoox"
    expected_output = "SSWWSSSWWS"
    run_pie_test_case("../p03798.py", input_content, expected_output)


def test_problem_p03798_2():
    input_content = "6\nooxoox"
    expected_output = "SSSWWS"
    run_pie_test_case("../p03798.py", input_content, expected_output)


def test_problem_p03798_3():
    input_content = "3\noox"
    expected_output = "-1"
    run_pie_test_case("../p03798.py", input_content, expected_output)
