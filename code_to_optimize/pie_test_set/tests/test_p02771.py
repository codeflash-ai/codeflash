from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02771_0():
    input_content = "5 7 5"
    expected_output = "Yes"
    run_pie_test_case("../p02771.py", input_content, expected_output)


def test_problem_p02771_1():
    input_content = "3 3 4"
    expected_output = "Yes"
    run_pie_test_case("../p02771.py", input_content, expected_output)


def test_problem_p02771_2():
    input_content = "4 4 4"
    expected_output = "No"
    run_pie_test_case("../p02771.py", input_content, expected_output)


def test_problem_p02771_3():
    input_content = "5 7 5"
    expected_output = "Yes"
    run_pie_test_case("../p02771.py", input_content, expected_output)


def test_problem_p02771_4():
    input_content = "4 9 6"
    expected_output = "No"
    run_pie_test_case("../p02771.py", input_content, expected_output)
