from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03220_0():
    input_content = "2\n12 5\n1000 2000"
    expected_output = "1"
    run_pie_test_case("../p03220.py", input_content, expected_output)
