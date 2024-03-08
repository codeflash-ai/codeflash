from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04045_0():
    input_content = "1000 8\n1 3 4 5 6 7 8 9"
    expected_output = "2000"
    run_pie_test_case("../p04045.py", input_content, expected_output)
