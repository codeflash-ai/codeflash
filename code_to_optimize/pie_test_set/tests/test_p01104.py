from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01104_0():
    input_content = "4 3\n110\n101\n011\n110\n7 1\n1\n1\n1\n1\n1\n1\n1\n4 5\n10000\n01000\n00100\n00010\n6 6\n111111\n011000\n100000\n000010\n100001\n100100\n0 0"
    expected_output = "3\n6\n0\n6"
    run_pie_test_case("../p01104.py", input_content, expected_output)


def test_problem_p01104_1():
    input_content = "4 3\n110\n101\n011\n110\n7 1\n1\n1\n1\n1\n1\n1\n1\n4 5\n10000\n01000\n00100\n00010\n6 6\n111111\n011000\n100000\n000010\n100001\n100100\n0 0"
    expected_output = "3\n6\n0\n6"
    run_pie_test_case("../p01104.py", input_content, expected_output)
