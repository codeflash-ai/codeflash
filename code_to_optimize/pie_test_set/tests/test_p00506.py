from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00506_0():
    input_content = "2\n75 125"
    expected_output = "1\n5\n25"
    run_pie_test_case("../p00506.py", input_content, expected_output)
