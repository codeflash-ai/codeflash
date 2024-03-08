from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03999_0():
    input_content = "125"
    expected_output = "176"
    run_pie_test_case("../p03999.py", input_content, expected_output)


def test_problem_p03999_1():
    input_content = "9999999999"
    expected_output = "12656242944"
    run_pie_test_case("../p03999.py", input_content, expected_output)


def test_problem_p03999_2():
    input_content = "125"
    expected_output = "176"
    run_pie_test_case("../p03999.py", input_content, expected_output)
