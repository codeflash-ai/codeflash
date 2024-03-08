from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00523_0():
    input_content = "6\n1\n5\n4\n5\n2\n4"
    expected_output = "6"
    run_pie_test_case("../p00523.py", input_content, expected_output)
