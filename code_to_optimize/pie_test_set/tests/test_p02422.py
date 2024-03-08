from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02422_0():
    input_content = "abcde\n3\nreplace 1 3 xyz\nreverse 0 2\nprint 1 4"
    expected_output = "xaze"
    run_pie_test_case("../p02422.py", input_content, expected_output)


def test_problem_p02422_1():
    input_content = "abcde\n3\nreplace 1 3 xyz\nreverse 0 2\nprint 1 4"
    expected_output = "xaze"
    run_pie_test_case("../p02422.py", input_content, expected_output)


def test_problem_p02422_2():
    input_content = "xyz\n3\nprint 0 2\nreplace 0 2 abc\nprint 0 2"
    expected_output = "xyz\nabc"
    run_pie_test_case("../p02422.py", input_content, expected_output)
