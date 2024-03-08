from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01202_0():
    input_content = "3\nUU\nRDUL\nULDURDULDURDULDURDULDURD"
    expected_output = "No\nYes\nYes"
    run_pie_test_case("../p01202.py", input_content, expected_output)


def test_problem_p01202_1():
    input_content = "3\nUU\nRDUL\nULDURDULDURDULDURDULDURD"
    expected_output = "No\nYes\nYes"
    run_pie_test_case("../p01202.py", input_content, expected_output)
