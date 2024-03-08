from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00488_0():
    input_content = "800\n700\n900\n198\n330"
    expected_output = "848"
    run_pie_test_case("../p00488.py", input_content, expected_output)


def test_problem_p00488_1():
    input_content = "800\n700\n900\n198\n330"
    expected_output = "848"
    run_pie_test_case("../p00488.py", input_content, expected_output)
