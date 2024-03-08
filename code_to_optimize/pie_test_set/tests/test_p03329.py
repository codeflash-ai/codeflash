from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03329_0():
    input_content = "127"
    expected_output = "4"
    run_pie_test_case("../p03329.py", input_content, expected_output)


def test_problem_p03329_1():
    input_content = "3"
    expected_output = "3"
    run_pie_test_case("../p03329.py", input_content, expected_output)


def test_problem_p03329_2():
    input_content = "44852"
    expected_output = "16"
    run_pie_test_case("../p03329.py", input_content, expected_output)


def test_problem_p03329_3():
    input_content = "127"
    expected_output = "4"
    run_pie_test_case("../p03329.py", input_content, expected_output)
