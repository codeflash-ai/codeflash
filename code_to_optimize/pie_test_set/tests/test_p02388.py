from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02388_0():
    input_content = "2"
    expected_output = "8"
    run_pie_test_case("../p02388.py", input_content, expected_output)
