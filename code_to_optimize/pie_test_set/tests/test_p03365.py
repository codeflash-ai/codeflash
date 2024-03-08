from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03365_0():
    input_content = "4"
    expected_output = "16"
    run_pie_test_case("../p03365.py", input_content, expected_output)


def test_problem_p03365_1():
    input_content = "2"
    expected_output = "1"
    run_pie_test_case("../p03365.py", input_content, expected_output)


def test_problem_p03365_2():
    input_content = "4"
    expected_output = "16"
    run_pie_test_case("../p03365.py", input_content, expected_output)


def test_problem_p03365_3():
    input_content = "5"
    expected_output = "84"
    run_pie_test_case("../p03365.py", input_content, expected_output)


def test_problem_p03365_4():
    input_content = "100000"
    expected_output = "341429644"
    run_pie_test_case("../p03365.py", input_content, expected_output)
