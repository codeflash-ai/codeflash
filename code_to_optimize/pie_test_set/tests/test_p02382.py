from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02382_0():
    input_content = "3\n1 2 3\n2 0 4"
    expected_output = "4.000000\n2.449490\n2.154435\n2.000000"
    run_pie_test_case("../p02382.py", input_content, expected_output)


def test_problem_p02382_1():
    input_content = "3\n1 2 3\n2 0 4"
    expected_output = "4.000000\n2.449490\n2.154435\n2.000000"
    run_pie_test_case("../p02382.py", input_content, expected_output)
