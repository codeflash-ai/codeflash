from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00065_0():
    input_content = "123,10\n56,12\n34,14\n\n123,3\n56,4\n123,5"
    expected_output = "56 2\n123 3"
    run_pie_test_case("../p00065.py", input_content, expected_output)


def test_problem_p00065_1():
    input_content = "123,10\n56,12\n34,14\n\n123,3\n56,4\n123,5"
    expected_output = "56 2\n123 3"
    run_pie_test_case("../p00065.py", input_content, expected_output)
