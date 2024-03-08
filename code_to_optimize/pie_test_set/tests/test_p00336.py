from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00336_0():
    input_content = "abab\nab"
    expected_output = "3"
    run_pie_test_case("../p00336.py", input_content, expected_output)


def test_problem_p00336_1():
    input_content = "data\nstructure"
    expected_output = "0"
    run_pie_test_case("../p00336.py", input_content, expected_output)


def test_problem_p00336_2():
    input_content = "aaaabaaaabaaaabaaaab\naaaaa"
    expected_output = "4368"
    run_pie_test_case("../p00336.py", input_content, expected_output)


def test_problem_p00336_3():
    input_content = "abab\nab"
    expected_output = "3"
    run_pie_test_case("../p00336.py", input_content, expected_output)
