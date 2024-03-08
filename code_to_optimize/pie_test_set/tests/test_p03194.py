from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03194_0():
    input_content = "3 24"
    expected_output = "2"
    run_pie_test_case("../p03194.py", input_content, expected_output)


def test_problem_p03194_1():
    input_content = "5 1"
    expected_output = "1"
    run_pie_test_case("../p03194.py", input_content, expected_output)


def test_problem_p03194_2():
    input_content = "4 972439611840"
    expected_output = "206"
    run_pie_test_case("../p03194.py", input_content, expected_output)


def test_problem_p03194_3():
    input_content = "3 24"
    expected_output = "2"
    run_pie_test_case("../p03194.py", input_content, expected_output)


def test_problem_p03194_4():
    input_content = "1 111"
    expected_output = "111"
    run_pie_test_case("../p03194.py", input_content, expected_output)
