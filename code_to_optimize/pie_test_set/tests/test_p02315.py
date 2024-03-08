from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02315_0():
    input_content = "4 5\n4 2\n5 2\n2 1\n8 3"
    expected_output = "13"
    run_pie_test_case("../p02315.py", input_content, expected_output)
