from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03919_0():
    input_content = "15 10\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snuke snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake\nsnake snake snake snake snake snake snake snake snake snake"
    expected_output = "H6"
    run_pie_test_case("../p03919.py", input_content, expected_output)
