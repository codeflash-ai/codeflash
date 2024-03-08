from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02667_0():
    input_content = "1101"
    expected_output = "5"
    run_pie_test_case("../p02667.py", input_content, expected_output)


def test_problem_p02667_1():
    input_content = "1101"
    expected_output = "5"
    run_pie_test_case("../p02667.py", input_content, expected_output)


def test_problem_p02667_2():
    input_content = "0111101101"
    expected_output = "26"
    run_pie_test_case("../p02667.py", input_content, expected_output)
