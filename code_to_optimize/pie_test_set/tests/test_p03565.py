from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03565_0():
    input_content = "?tc????\ncoder"
    expected_output = "atcoder"
    run_pie_test_case("../p03565.py", input_content, expected_output)


def test_problem_p03565_1():
    input_content = "?tc????\ncoder"
    expected_output = "atcoder"
    run_pie_test_case("../p03565.py", input_content, expected_output)


def test_problem_p03565_2():
    input_content = "??p??d??\nabc"
    expected_output = "UNRESTORABLE"
    run_pie_test_case("../p03565.py", input_content, expected_output)
