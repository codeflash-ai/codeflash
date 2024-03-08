from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03231_0():
    input_content = "3 2\nacp\nae"
    expected_output = "6"
    run_pie_test_case("../p03231.py", input_content, expected_output)


def test_problem_p03231_1():
    input_content = "3 2\nacp\nae"
    expected_output = "6"
    run_pie_test_case("../p03231.py", input_content, expected_output)


def test_problem_p03231_2():
    input_content = "6 3\nabcdef\nabc"
    expected_output = "-1"
    run_pie_test_case("../p03231.py", input_content, expected_output)


def test_problem_p03231_3():
    input_content = "15 9\ndnsusrayukuaiia\ndujrunuma"
    expected_output = "45"
    run_pie_test_case("../p03231.py", input_content, expected_output)
