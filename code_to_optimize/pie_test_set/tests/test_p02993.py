from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02993_0():
    input_content = "3776"
    expected_output = "Bad"
    run_pie_test_case("../p02993.py", input_content, expected_output)


def test_problem_p02993_1():
    input_content = "1333"
    expected_output = "Bad"
    run_pie_test_case("../p02993.py", input_content, expected_output)


def test_problem_p02993_2():
    input_content = "8080"
    expected_output = "Good"
    run_pie_test_case("../p02993.py", input_content, expected_output)


def test_problem_p02993_3():
    input_content = "3776"
    expected_output = "Bad"
    run_pie_test_case("../p02993.py", input_content, expected_output)


def test_problem_p02993_4():
    input_content = "0024"
    expected_output = "Bad"
    run_pie_test_case("../p02993.py", input_content, expected_output)
