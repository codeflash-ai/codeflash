from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03854_0():
    input_content = "erasedream"
    expected_output = "YES"
    run_pie_test_case("../p03854.py", input_content, expected_output)


def test_problem_p03854_1():
    input_content = "dreamerer"
    expected_output = "NO"
    run_pie_test_case("../p03854.py", input_content, expected_output)


def test_problem_p03854_2():
    input_content = "erasedream"
    expected_output = "YES"
    run_pie_test_case("../p03854.py", input_content, expected_output)


def test_problem_p03854_3():
    input_content = "dreameraser"
    expected_output = "YES"
    run_pie_test_case("../p03854.py", input_content, expected_output)
