from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00011_0():
    input_content = "5\n4\n2,4\n3,5\n1,2\n3,4"
    expected_output = "4\n1\n2\n5\n3"
    run_pie_test_case("../p00011.py", input_content, expected_output)


def test_problem_p00011_1():
    input_content = "5\n4\n2,4\n3,5\n1,2\n3,4"
    expected_output = "4\n1\n2\n5\n3"
    run_pie_test_case("../p00011.py", input_content, expected_output)
