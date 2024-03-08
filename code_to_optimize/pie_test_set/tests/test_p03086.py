from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03086_0():
    input_content = "ATCODER"
    expected_output = "3"
    run_pie_test_case("../p03086.py", input_content, expected_output)


def test_problem_p03086_1():
    input_content = "SHINJUKU"
    expected_output = "0"
    run_pie_test_case("../p03086.py", input_content, expected_output)


def test_problem_p03086_2():
    input_content = "ATCODER"
    expected_output = "3"
    run_pie_test_case("../p03086.py", input_content, expected_output)


def test_problem_p03086_3():
    input_content = "HATAGAYA"
    expected_output = "5"
    run_pie_test_case("../p03086.py", input_content, expected_output)
