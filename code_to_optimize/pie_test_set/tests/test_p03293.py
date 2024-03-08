from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03293_0():
    input_content = "kyoto\ntokyo"
    expected_output = "Yes"
    run_pie_test_case("../p03293.py", input_content, expected_output)


def test_problem_p03293_1():
    input_content = "aaaaaaaaaaaaaaab\naaaaaaaaaaaaaaab"
    expected_output = "Yes"
    run_pie_test_case("../p03293.py", input_content, expected_output)


def test_problem_p03293_2():
    input_content = "kyoto\ntokyo"
    expected_output = "Yes"
    run_pie_test_case("../p03293.py", input_content, expected_output)


def test_problem_p03293_3():
    input_content = "abc\narc"
    expected_output = "No"
    run_pie_test_case("../p03293.py", input_content, expected_output)
