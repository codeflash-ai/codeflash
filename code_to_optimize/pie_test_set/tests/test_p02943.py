from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02943_0():
    input_content = "5 1\nbacba"
    expected_output = "aabca"
    run_pie_test_case("../p02943.py", input_content, expected_output)


def test_problem_p02943_1():
    input_content = "5 1\nbacba"
    expected_output = "aabca"
    run_pie_test_case("../p02943.py", input_content, expected_output)


def test_problem_p02943_2():
    input_content = "10 2\nbbaabbbaab"
    expected_output = "aaaabbaabb"
    run_pie_test_case("../p02943.py", input_content, expected_output)
