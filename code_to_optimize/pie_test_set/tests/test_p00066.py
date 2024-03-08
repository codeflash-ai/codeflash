from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00066_0():
    input_content = "ooosxssxs\nxoosxsosx\nooxxxooxo"
    expected_output = "o\nx\nd"
    run_pie_test_case("../p00066.py", input_content, expected_output)


def test_problem_p00066_1():
    input_content = "ooosxssxs\nxoosxsosx\nooxxxooxo"
    expected_output = "o\nx\nd"
    run_pie_test_case("../p00066.py", input_content, expected_output)
