from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03564_0():
    input_content = "4\n3"
    expected_output = "10"
    run_pie_test_case("../p03564.py", input_content, expected_output)


def test_problem_p03564_1():
    input_content = "10\n10"
    expected_output = "76"
    run_pie_test_case("../p03564.py", input_content, expected_output)


def test_problem_p03564_2():
    input_content = "4\n3"
    expected_output = "10"
    run_pie_test_case("../p03564.py", input_content, expected_output)
