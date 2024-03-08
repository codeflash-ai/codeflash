from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03563_0():
    input_content = "2002\n2017"
    expected_output = "2032"
    run_pie_test_case("../p03563.py", input_content, expected_output)


def test_problem_p03563_1():
    input_content = "4500\n0"
    expected_output = "-4500"
    run_pie_test_case("../p03563.py", input_content, expected_output)


def test_problem_p03563_2():
    input_content = "2002\n2017"
    expected_output = "2032"
    run_pie_test_case("../p03563.py", input_content, expected_output)
