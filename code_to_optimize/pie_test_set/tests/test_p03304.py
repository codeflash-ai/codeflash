from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03304_0():
    input_content = "2 3 1"
    expected_output = "1.0000000000"
    run_pie_test_case("../p03304.py", input_content, expected_output)


def test_problem_p03304_1():
    input_content = "2 3 1"
    expected_output = "1.0000000000"
    run_pie_test_case("../p03304.py", input_content, expected_output)


def test_problem_p03304_2():
    input_content = "1000000000 180707 0"
    expected_output = "0.0001807060"
    run_pie_test_case("../p03304.py", input_content, expected_output)
