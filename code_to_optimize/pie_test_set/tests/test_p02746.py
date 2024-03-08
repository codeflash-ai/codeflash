from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02746_0():
    input_content = "2\n4 2 7 4\n9 9 1 9"
    expected_output = "5\n8"
    run_pie_test_case("../p02746.py", input_content, expected_output)


def test_problem_p02746_1():
    input_content = "2\n4 2 7 4\n9 9 1 9"
    expected_output = "5\n8"
    run_pie_test_case("../p02746.py", input_content, expected_output)
