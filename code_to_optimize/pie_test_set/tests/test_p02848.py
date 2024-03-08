from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02848_0():
    input_content = "2\nABCXYZ"
    expected_output = "CDEZAB"
    run_pie_test_case("../p02848.py", input_content, expected_output)


def test_problem_p02848_1():
    input_content = "13\nABCDEFGHIJKLMNOPQRSTUVWXYZ"
    expected_output = "NOPQRSTUVWXYZABCDEFGHIJKLM"
    run_pie_test_case("../p02848.py", input_content, expected_output)


def test_problem_p02848_2():
    input_content = "0\nABCXYZ"
    expected_output = "ABCXYZ"
    run_pie_test_case("../p02848.py", input_content, expected_output)


def test_problem_p02848_3():
    input_content = "2\nABCXYZ"
    expected_output = "CDEZAB"
    run_pie_test_case("../p02848.py", input_content, expected_output)
