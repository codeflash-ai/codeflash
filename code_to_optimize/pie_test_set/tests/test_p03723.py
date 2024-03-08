from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03723_0():
    input_content = "4 12 20"
    expected_output = "3"
    run_pie_test_case("../p03723.py", input_content, expected_output)


def test_problem_p03723_1():
    input_content = "14 14 14"
    expected_output = "-1"
    run_pie_test_case("../p03723.py", input_content, expected_output)


def test_problem_p03723_2():
    input_content = "454 414 444"
    expected_output = "1"
    run_pie_test_case("../p03723.py", input_content, expected_output)


def test_problem_p03723_3():
    input_content = "4 12 20"
    expected_output = "3"
    run_pie_test_case("../p03723.py", input_content, expected_output)
