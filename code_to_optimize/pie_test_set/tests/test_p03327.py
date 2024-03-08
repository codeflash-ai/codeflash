from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03327_0():
    input_content = "999"
    expected_output = "ABC"
    run_pie_test_case("../p03327.py", input_content, expected_output)


def test_problem_p03327_1():
    input_content = "1000"
    expected_output = "ABD"
    run_pie_test_case("../p03327.py", input_content, expected_output)


def test_problem_p03327_2():
    input_content = "1481"
    expected_output = "ABD"
    run_pie_test_case("../p03327.py", input_content, expected_output)


def test_problem_p03327_3():
    input_content = "999"
    expected_output = "ABC"
    run_pie_test_case("../p03327.py", input_content, expected_output)
