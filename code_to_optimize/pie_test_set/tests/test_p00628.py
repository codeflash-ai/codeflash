from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00628_0():
    input_content = "Yes I have a number\nHow I wish I could calculate an unused color for space\nThank you\nEND OF INPUT"
    expected_output = "31416\n31415926535\n53"
    run_pie_test_case("../p00628.py", input_content, expected_output)


def test_problem_p00628_1():
    input_content = "Yes I have a number\nHow I wish I could calculate an unused color for space\nThank you\nEND OF INPUT"
    expected_output = "31416\n31415926535\n53"
    run_pie_test_case("../p00628.py", input_content, expected_output)
