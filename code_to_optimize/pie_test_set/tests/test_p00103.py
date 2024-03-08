from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00103_0():
    input_content = "2\nHIT\nOUT\nHOMERUN\nHIT\nHIT\nHOMERUN\nHIT\nOUT\nHIT\nHIT\nHIT\nHIT\nOUT\nHIT\nHIT\nOUT\nHIT\nOUT\nOUT"
    expected_output = "7\n0"
    run_pie_test_case("../p00103.py", input_content, expected_output)


def test_problem_p00103_1():
    input_content = "2\nHIT\nOUT\nHOMERUN\nHIT\nHIT\nHOMERUN\nHIT\nOUT\nHIT\nHIT\nHIT\nHIT\nOUT\nHIT\nHIT\nOUT\nHIT\nOUT\nOUT"
    expected_output = "7\n0"
    run_pie_test_case("../p00103.py", input_content, expected_output)
