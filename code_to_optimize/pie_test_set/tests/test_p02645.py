from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02645_0():
    input_content = "takahashi"
    expected_output = "tak"
    run_pie_test_case("../p02645.py", input_content, expected_output)


def test_problem_p02645_1():
    input_content = "naohiro"
    expected_output = "nao"
    run_pie_test_case("../p02645.py", input_content, expected_output)


def test_problem_p02645_2():
    input_content = "takahashi"
    expected_output = "tak"
    run_pie_test_case("../p02645.py", input_content, expected_output)
