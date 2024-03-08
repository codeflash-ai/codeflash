from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00183_0():
    input_content = "bbw\nwbw\n+b+\nbwb\nwbw\nwbw\n0"
    expected_output = "b\nNA"
    run_pie_test_case("../p00183.py", input_content, expected_output)


def test_problem_p00183_1():
    input_content = "bbw\nwbw\n+b+\nbwb\nwbw\nwbw\n0"
    expected_output = "b\nNA"
    run_pie_test_case("../p00183.py", input_content, expected_output)
