from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02836_0():
    input_content = "redcoder"
    expected_output = "1"
    run_pie_test_case("../p02836.py", input_content, expected_output)


def test_problem_p02836_1():
    input_content = "abcdabc"
    expected_output = "2"
    run_pie_test_case("../p02836.py", input_content, expected_output)


def test_problem_p02836_2():
    input_content = "redcoder"
    expected_output = "1"
    run_pie_test_case("../p02836.py", input_content, expected_output)


def test_problem_p02836_3():
    input_content = "vvvvvv"
    expected_output = "0"
    run_pie_test_case("../p02836.py", input_content, expected_output)
