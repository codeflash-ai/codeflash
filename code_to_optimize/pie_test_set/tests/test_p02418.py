from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02418_0():
    input_content = "vanceknowledgetoad\nadvance"
    expected_output = "Yes"
    run_pie_test_case("../p02418.py", input_content, expected_output)


def test_problem_p02418_1():
    input_content = "vanceknowledgetoad\nadvance"
    expected_output = "Yes"
    run_pie_test_case("../p02418.py", input_content, expected_output)


def test_problem_p02418_2():
    input_content = "vanceknowledgetoad\nadvanced"
    expected_output = "No"
    run_pie_test_case("../p02418.py", input_content, expected_output)
