from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02402_0():
    input_content = "5\n10 1 5 4 17"
    expected_output = "1 17 37"
    run_pie_test_case("../p02402.py", input_content, expected_output)


def test_problem_p02402_1():
    input_content = "5\n10 1 5 4 17"
    expected_output = "1 17 37"
    run_pie_test_case("../p02402.py", input_content, expected_output)
