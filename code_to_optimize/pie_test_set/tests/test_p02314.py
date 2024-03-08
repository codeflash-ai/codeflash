from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02314_0():
    input_content = "55 4\n1 5 10 50"
    expected_output = "2"
    run_pie_test_case("../p02314.py", input_content, expected_output)
