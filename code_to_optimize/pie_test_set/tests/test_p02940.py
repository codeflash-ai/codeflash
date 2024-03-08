from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02940_0():
    input_content = "3\nRRRGGGBBB"
    expected_output = "216"
    run_pie_test_case("../p02940.py", input_content, expected_output)


def test_problem_p02940_1():
    input_content = "3\nRRRGGGBBB"
    expected_output = "216"
    run_pie_test_case("../p02940.py", input_content, expected_output)


def test_problem_p02940_2():
    input_content = "5\nBBRGRRGRGGRBBGB"
    expected_output = "960"
    run_pie_test_case("../p02940.py", input_content, expected_output)
