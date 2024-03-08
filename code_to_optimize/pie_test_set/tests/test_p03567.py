from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03567_0():
    input_content = "BACD"
    expected_output = "Yes"
    run_pie_test_case("../p03567.py", input_content, expected_output)


def test_problem_p03567_1():
    input_content = "CABD"
    expected_output = "No"
    run_pie_test_case("../p03567.py", input_content, expected_output)


def test_problem_p03567_2():
    input_content = "BACD"
    expected_output = "Yes"
    run_pie_test_case("../p03567.py", input_content, expected_output)


def test_problem_p03567_3():
    input_content = "ACACA"
    expected_output = "Yes"
    run_pie_test_case("../p03567.py", input_content, expected_output)


def test_problem_p03567_4():
    input_content = "ABCD"
    expected_output = "No"
    run_pie_test_case("../p03567.py", input_content, expected_output)


def test_problem_p03567_5():
    input_content = "XX"
    expected_output = "No"
    run_pie_test_case("../p03567.py", input_content, expected_output)
