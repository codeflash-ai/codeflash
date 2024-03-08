from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02275_0():
    input_content = "7\n2 5 1 3 2 3 0"
    expected_output = "0 1 2 2 3 3 5"
    run_pie_test_case("../p02275.py", input_content, expected_output)


def test_problem_p02275_1():
    input_content = "7\n2 5 1 3 2 3 0"
    expected_output = "0 1 2 2 3 3 5"
    run_pie_test_case("../p02275.py", input_content, expected_output)
