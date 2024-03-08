from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02415_0():
    input_content = "fAIR, LATER, OCCASIONALLY CLOUDY."
    expected_output = "Fair, later, occasionally cloudy."
    run_pie_test_case("../p02415.py", input_content, expected_output)


def test_problem_p02415_1():
    input_content = "fAIR, LATER, OCCASIONALLY CLOUDY."
    expected_output = "Fair, later, occasionally cloudy."
    run_pie_test_case("../p02415.py", input_content, expected_output)
