from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03624_0():
    input_content = "atcoderregularcontest"
    expected_output = "b"
    run_pie_test_case("../p03624.py", input_content, expected_output)
