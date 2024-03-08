from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03035_0():
    input_content = "30 100"
    expected_output = "100"
    run_pie_test_case("../p03035.py", input_content, expected_output)


def test_problem_p03035_1():
    input_content = "0 100"
    expected_output = "0"
    run_pie_test_case("../p03035.py", input_content, expected_output)


def test_problem_p03035_2():
    input_content = "30 100"
    expected_output = "100"
    run_pie_test_case("../p03035.py", input_content, expected_output)


def test_problem_p03035_3():
    input_content = "12 100"
    expected_output = "50"
    run_pie_test_case("../p03035.py", input_content, expected_output)
