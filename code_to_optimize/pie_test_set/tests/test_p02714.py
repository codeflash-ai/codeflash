from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02714_0():
    input_content = "4\nRRGB"
    expected_output = "1"
    run_pie_test_case("../p02714.py", input_content, expected_output)


def test_problem_p02714_1():
    input_content = "4\nRRGB"
    expected_output = "1"
    run_pie_test_case("../p02714.py", input_content, expected_output)


def test_problem_p02714_2():
    input_content = "39\nRBRBGRBGGBBRRGBBRRRBGGBRBGBRBGBRBBBGBBB"
    expected_output = "1800"
    run_pie_test_case("../p02714.py", input_content, expected_output)
