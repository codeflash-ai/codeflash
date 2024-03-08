from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02639_0():
    input_content = "0 2 3 4 5"
    expected_output = "1"
    run_pie_test_case("../p02639.py", input_content, expected_output)
