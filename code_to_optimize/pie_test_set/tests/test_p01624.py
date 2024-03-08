from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01624_0():
    input_content = "1 1\n2 2\n3 1+2*3-4\n3 1|2^3&4\n3 (1+2)*3\n3 1-1-1-1\n0 #"
    expected_output = "91\n2\n273\n93\n279\n88"
    run_pie_test_case("../p01624.py", input_content, expected_output)


def test_problem_p01624_1():
    input_content = "1 1\n2 2\n3 1+2*3-4\n3 1|2^3&4\n3 (1+2)*3\n3 1-1-1-1\n0 #"
    expected_output = "91\n2\n273\n93\n279\n88"
    run_pie_test_case("../p01624.py", input_content, expected_output)
