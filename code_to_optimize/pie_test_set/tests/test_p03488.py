from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03488_0():
    input_content = "FTFFTFFF\n4 2"
    expected_output = "Yes"
    run_pie_test_case("../p03488.py", input_content, expected_output)
