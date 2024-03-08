from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02945_0():
    input_content = "-13 3"
    expected_output = "-10"
    run_pie_test_case("../p02945.py", input_content, expected_output)
