from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03605_0():
    input_content = "29"
    expected_output = "Yes"
    run_pie_test_case("../p03605.py", input_content, expected_output)


def test_problem_p03605_1():
    input_content = "29"
    expected_output = "Yes"
    run_pie_test_case("../p03605.py", input_content, expected_output)


def test_problem_p03605_2():
    input_content = "91"
    expected_output = "Yes"
    run_pie_test_case("../p03605.py", input_content, expected_output)


def test_problem_p03605_3():
    input_content = "72"
    expected_output = "No"
    run_pie_test_case("../p03605.py", input_content, expected_output)
