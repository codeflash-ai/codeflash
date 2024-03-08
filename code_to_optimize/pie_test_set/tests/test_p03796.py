from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03796_0():
    input_content = "3"
    expected_output = "6"
    run_pie_test_case("../p03796.py", input_content, expected_output)


def test_problem_p03796_1():
    input_content = "100000"
    expected_output = "457992974"
    run_pie_test_case("../p03796.py", input_content, expected_output)


def test_problem_p03796_2():
    input_content = "3"
    expected_output = "6"
    run_pie_test_case("../p03796.py", input_content, expected_output)


def test_problem_p03796_3():
    input_content = "10"
    expected_output = "3628800"
    run_pie_test_case("../p03796.py", input_content, expected_output)
