from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03063_0():
    input_content = "3\n#.#"
    expected_output = "1"
    run_pie_test_case("../p03063.py", input_content, expected_output)
