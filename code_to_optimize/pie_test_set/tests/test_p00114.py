from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00114_0():
    input_content = "2 5 3 7 6 13\n517 1024 746 6561 4303 3125\n0 0 0 0 0 0"
    expected_output = "12\n116640000"
    run_pie_test_case("../p00114.py", input_content, expected_output)


def test_problem_p00114_1():
    input_content = "2 5 3 7 6 13\n517 1024 746 6561 4303 3125\n0 0 0 0 0 0"
    expected_output = "12\n116640000"
    run_pie_test_case("../p00114.py", input_content, expected_output)
