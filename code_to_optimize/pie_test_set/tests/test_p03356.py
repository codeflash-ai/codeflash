from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03356_0():
    input_content = "5 2\n5 3 1 4 2\n1 3\n5 4"
    expected_output = "2"
    run_pie_test_case("../p03356.py", input_content, expected_output)
