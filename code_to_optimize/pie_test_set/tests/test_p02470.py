from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02470_0():
    input_content = "6"
    expected_output = "2"
    run_pie_test_case("../p02470.py", input_content, expected_output)


def test_problem_p02470_1():
    input_content = "1000000"
    expected_output = "400000"
    run_pie_test_case("../p02470.py", input_content, expected_output)


def test_problem_p02470_2():
    input_content = "6"
    expected_output = "2"
    run_pie_test_case("../p02470.py", input_content, expected_output)
