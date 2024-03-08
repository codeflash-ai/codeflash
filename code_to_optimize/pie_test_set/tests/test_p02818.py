from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02818_0():
    input_content = "2 3 3"
    expected_output = "0 2"
    run_pie_test_case("../p02818.py", input_content, expected_output)


def test_problem_p02818_1():
    input_content = "500000000000 500000000000 1000000000000"
    expected_output = "0 0"
    run_pie_test_case("../p02818.py", input_content, expected_output)


def test_problem_p02818_2():
    input_content = "2 3 3"
    expected_output = "0 2"
    run_pie_test_case("../p02818.py", input_content, expected_output)
