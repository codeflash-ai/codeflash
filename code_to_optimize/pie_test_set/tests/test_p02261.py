from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02261_0():
    input_content = "5\nH4 C9 S4 D2 C3"
    expected_output = "D2 C3 H4 S4 C9\nStable\nD2 C3 S4 H4 C9\nNot stable"
    run_pie_test_case("../p02261.py", input_content, expected_output)


def test_problem_p02261_1():
    input_content = "2\nS1 H1"
    expected_output = "S1 H1\nStable\nS1 H1\nStable"
    run_pie_test_case("../p02261.py", input_content, expected_output)


def test_problem_p02261_2():
    input_content = "5\nH4 C9 S4 D2 C3"
    expected_output = "D2 C3 H4 S4 C9\nStable\nD2 C3 S4 H4 C9\nNot stable"
    run_pie_test_case("../p02261.py", input_content, expected_output)
