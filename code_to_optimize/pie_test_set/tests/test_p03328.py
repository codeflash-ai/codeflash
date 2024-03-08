from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03328_0():
    input_content = "8 13"
    expected_output = "2"
    run_pie_test_case("../p03328.py", input_content, expected_output)


def test_problem_p03328_1():
    input_content = "54 65"
    expected_output = "1"
    run_pie_test_case("../p03328.py", input_content, expected_output)


def test_problem_p03328_2():
    input_content = "8 13"
    expected_output = "2"
    run_pie_test_case("../p03328.py", input_content, expected_output)
