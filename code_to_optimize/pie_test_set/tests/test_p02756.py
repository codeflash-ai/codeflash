from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02756_0():
    input_content = "a\n4\n2 1 p\n1\n2 2 c\n1"
    expected_output = "cpa"
    run_pie_test_case("../p02756.py", input_content, expected_output)


def test_problem_p02756_1():
    input_content = "y\n1\n2 1 x"
    expected_output = "xy"
    run_pie_test_case("../p02756.py", input_content, expected_output)


def test_problem_p02756_2():
    input_content = "a\n4\n2 1 p\n1\n2 2 c\n1"
    expected_output = "cpa"
    run_pie_test_case("../p02756.py", input_content, expected_output)


def test_problem_p02756_3():
    input_content = "a\n6\n2 2 a\n2 1 b\n1\n2 2 c\n1\n1"
    expected_output = "aabc"
    run_pie_test_case("../p02756.py", input_content, expected_output)
