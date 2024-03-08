from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03614_0():
    input_content = "5\n1 4 3 5 2"
    expected_output = "2"
    run_pie_test_case("../p03614.py", input_content, expected_output)
