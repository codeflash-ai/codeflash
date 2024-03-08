from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02571_0():
    input_content = "cabacc\nabc"
    expected_output = "1"
    run_pie_test_case("../p02571.py", input_content, expected_output)


def test_problem_p02571_1():
    input_content = "codeforces\natcoder"
    expected_output = "6"
    run_pie_test_case("../p02571.py", input_content, expected_output)


def test_problem_p02571_2():
    input_content = "cabacc\nabc"
    expected_output = "1"
    run_pie_test_case("../p02571.py", input_content, expected_output)
