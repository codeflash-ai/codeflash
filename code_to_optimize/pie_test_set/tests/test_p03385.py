from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03385_0():
    input_content = "bac"
    expected_output = "Yes"
    run_pie_test_case("../p03385.py", input_content, expected_output)


def test_problem_p03385_1():
    input_content = "bab"
    expected_output = "No"
    run_pie_test_case("../p03385.py", input_content, expected_output)


def test_problem_p03385_2():
    input_content = "abc"
    expected_output = "Yes"
    run_pie_test_case("../p03385.py", input_content, expected_output)


def test_problem_p03385_3():
    input_content = "aaa"
    expected_output = "No"
    run_pie_test_case("../p03385.py", input_content, expected_output)


def test_problem_p03385_4():
    input_content = "bac"
    expected_output = "Yes"
    run_pie_test_case("../p03385.py", input_content, expected_output)
