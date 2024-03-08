from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00445_0():
    input_content = "JOIJOI\nJOIOIOIOI\nJOIOIJOINXNXJIOIOIOJ"
    expected_output = "2\n0\n1\n3\n2\n3"
    run_pie_test_case("../p00445.py", input_content, expected_output)


def test_problem_p00445_1():
    input_content = "JOIJOI\nJOIOIOIOI\nJOIOIJOINXNXJIOIOIOJ"
    expected_output = "2\n0\n1\n3\n2\n3"
    run_pie_test_case("../p00445.py", input_content, expected_output)


def test_problem_p00445_2():
    input_content = "None"
    expected_output = "None"
    run_pie_test_case("../p00445.py", input_content, expected_output)
