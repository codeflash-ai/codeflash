from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03763_0():
    input_content = "3\ncbaa\ndaacc\nacacac"
    expected_output = "aac"
    run_pie_test_case("../p03763.py", input_content, expected_output)
