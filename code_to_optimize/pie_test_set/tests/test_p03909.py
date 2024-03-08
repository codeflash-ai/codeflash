from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03909_0():
    input_content = "15 10\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snuke snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake"
    expected_output = "H6"
    run_pie_test_case("../p03909.py", input_content, expected_output)


def test_problem_p03909_1():
    input_content = "15 10\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snuke snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake"
    expected_output = "H6"
    run_pie_test_case("../p03909.py", input_content, expected_output)


def test_problem_p03909_2():
    input_content = "1 1\nsnuke"
    expected_output = "A1"
    run_pie_test_case("../p03909.py", input_content, expected_output)
