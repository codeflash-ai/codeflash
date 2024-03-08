from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03601_0():
    input_content = "1 2 10 20 15 200"
    expected_output = "110 10"
    run_pie_test_case("../p03601.py", input_content, expected_output)
