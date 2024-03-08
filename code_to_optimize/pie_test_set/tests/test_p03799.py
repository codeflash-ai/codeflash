from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03799_0():
    input_content = "1 6"
    expected_output = "2"
    run_pie_test_case("../p03799.py", input_content, expected_output)


def test_problem_p03799_1():
    input_content = "12345 678901"
    expected_output = "175897"
    run_pie_test_case("../p03799.py", input_content, expected_output)


def test_problem_p03799_2():
    input_content = "1 6"
    expected_output = "2"
    run_pie_test_case("../p03799.py", input_content, expected_output)
