from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00039_0():
    input_content = "IV\nCCCCLXXXXVIIII\nCDXCIX"
    expected_output = "4\n499\n499"
    run_pie_test_case("../p00039.py", input_content, expected_output)


def test_problem_p00039_1():
    input_content = "IV\nCCCCLXXXXVIIII\nCDXCIX"
    expected_output = "4\n499\n499"
    run_pie_test_case("../p00039.py", input_content, expected_output)
