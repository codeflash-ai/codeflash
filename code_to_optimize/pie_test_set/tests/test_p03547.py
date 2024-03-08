from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03547_0():
    input_content = "A B"
    expected_output = "<"
    run_pie_test_case("../p03547.py", input_content, expected_output)


def test_problem_p03547_1():
    input_content = "F F"
    expected_output = "="
    run_pie_test_case("../p03547.py", input_content, expected_output)


def test_problem_p03547_2():
    input_content = "A B"
    expected_output = "<"
    run_pie_test_case("../p03547.py", input_content, expected_output)


def test_problem_p03547_3():
    input_content = "E C"
    expected_output = ">"
    run_pie_test_case("../p03547.py", input_content, expected_output)
