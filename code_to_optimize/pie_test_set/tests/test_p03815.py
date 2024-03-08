from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03815_0():
    input_content = "7"
    expected_output = "2"
    run_pie_test_case("../p03815.py", input_content, expected_output)


def test_problem_p03815_1():
    input_content = "7"
    expected_output = "2"
    run_pie_test_case("../p03815.py", input_content, expected_output)


def test_problem_p03815_2():
    input_content = "149696127901"
    expected_output = "27217477801"
    run_pie_test_case("../p03815.py", input_content, expected_output)
