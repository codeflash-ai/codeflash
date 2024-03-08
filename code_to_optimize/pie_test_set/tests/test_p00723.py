from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00723_0():
    input_content = "4\naa\nabba\nabcd\nabcde"
    expected_output = "1\n6\n12\n18"
    run_pie_test_case("../p00723.py", input_content, expected_output)


def test_problem_p00723_1():
    input_content = "4\naa\nabba\nabcd\nabcde"
    expected_output = "1\n6\n12\n18"
    run_pie_test_case("../p00723.py", input_content, expected_output)
