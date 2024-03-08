from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03242_0():
    input_content = "119"
    expected_output = "991"
    run_pie_test_case("../p03242.py", input_content, expected_output)


def test_problem_p03242_1():
    input_content = "119"
    expected_output = "991"
    run_pie_test_case("../p03242.py", input_content, expected_output)


def test_problem_p03242_2():
    input_content = "999"
    expected_output = "111"
    run_pie_test_case("../p03242.py", input_content, expected_output)
