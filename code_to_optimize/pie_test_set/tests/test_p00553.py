from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00553_0():
    input_content = "-10\n20\n5\n10\n3"
    expected_output = "120"
    run_pie_test_case("../p00553.py", input_content, expected_output)
