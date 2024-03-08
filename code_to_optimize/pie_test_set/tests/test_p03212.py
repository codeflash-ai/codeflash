from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03212_0():
    input_content = "575"
    expected_output = "4"
    run_pie_test_case("../p03212.py", input_content, expected_output)


def test_problem_p03212_1():
    input_content = "575"
    expected_output = "4"
    run_pie_test_case("../p03212.py", input_content, expected_output)


def test_problem_p03212_2():
    input_content = "3600"
    expected_output = "13"
    run_pie_test_case("../p03212.py", input_content, expected_output)


def test_problem_p03212_3():
    input_content = "999999999"
    expected_output = "26484"
    run_pie_test_case("../p03212.py", input_content, expected_output)
