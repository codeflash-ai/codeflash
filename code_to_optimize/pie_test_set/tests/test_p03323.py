from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03323_0():
    input_content = "5 4"
    expected_output = "Yay!"
    run_pie_test_case("../p03323.py", input_content, expected_output)


def test_problem_p03323_1():
    input_content = "11 4"
    expected_output = ":("
    run_pie_test_case("../p03323.py", input_content, expected_output)


def test_problem_p03323_2():
    input_content = "8 8"
    expected_output = "Yay!"
    run_pie_test_case("../p03323.py", input_content, expected_output)


def test_problem_p03323_3():
    input_content = "5 4"
    expected_output = "Yay!"
    run_pie_test_case("../p03323.py", input_content, expected_output)
