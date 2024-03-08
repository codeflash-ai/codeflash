from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03071_0():
    input_content = "5 3"
    expected_output = "9"
    run_pie_test_case("../p03071.py", input_content, expected_output)


def test_problem_p03071_1():
    input_content = "3 4"
    expected_output = "7"
    run_pie_test_case("../p03071.py", input_content, expected_output)


def test_problem_p03071_2():
    input_content = "5 3"
    expected_output = "9"
    run_pie_test_case("../p03071.py", input_content, expected_output)


def test_problem_p03071_3():
    input_content = "6 6"
    expected_output = "12"
    run_pie_test_case("../p03071.py", input_content, expected_output)
