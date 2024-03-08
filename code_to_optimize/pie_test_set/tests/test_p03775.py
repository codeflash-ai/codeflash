from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03775_0():
    input_content = "10000"
    expected_output = "3"
    run_pie_test_case("../p03775.py", input_content, expected_output)


def test_problem_p03775_1():
    input_content = "1000003"
    expected_output = "7"
    run_pie_test_case("../p03775.py", input_content, expected_output)


def test_problem_p03775_2():
    input_content = "10000"
    expected_output = "3"
    run_pie_test_case("../p03775.py", input_content, expected_output)


def test_problem_p03775_3():
    input_content = "9876543210"
    expected_output = "6"
    run_pie_test_case("../p03775.py", input_content, expected_output)
