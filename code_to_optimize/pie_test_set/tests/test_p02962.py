from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02962_0():
    input_content = "abcabab\nab"
    expected_output = "3"
    run_pie_test_case("../p02962.py", input_content, expected_output)


def test_problem_p02962_1():
    input_content = "aa\naaaaaaa"
    expected_output = "-1"
    run_pie_test_case("../p02962.py", input_content, expected_output)


def test_problem_p02962_2():
    input_content = "aba\nbaaab"
    expected_output = "0"
    run_pie_test_case("../p02962.py", input_content, expected_output)


def test_problem_p02962_3():
    input_content = "abcabab\nab"
    expected_output = "3"
    run_pie_test_case("../p02962.py", input_content, expected_output)
