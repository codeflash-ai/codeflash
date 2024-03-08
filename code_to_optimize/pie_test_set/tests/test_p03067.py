from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03067_0():
    input_content = "3 8 5"
    expected_output = "Yes"
    run_pie_test_case("../p03067.py", input_content, expected_output)


def test_problem_p03067_1():
    input_content = "10 2 4"
    expected_output = "Yes"
    run_pie_test_case("../p03067.py", input_content, expected_output)


def test_problem_p03067_2():
    input_content = "7 3 1"
    expected_output = "No"
    run_pie_test_case("../p03067.py", input_content, expected_output)


def test_problem_p03067_3():
    input_content = "31 41 59"
    expected_output = "No"
    run_pie_test_case("../p03067.py", input_content, expected_output)


def test_problem_p03067_4():
    input_content = "3 8 5"
    expected_output = "Yes"
    run_pie_test_case("../p03067.py", input_content, expected_output)
