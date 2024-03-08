from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02711_0():
    input_content = "117"
    expected_output = "Yes"
    run_pie_test_case("../p02711.py", input_content, expected_output)


def test_problem_p02711_1():
    input_content = "777"
    expected_output = "Yes"
    run_pie_test_case("../p02711.py", input_content, expected_output)


def test_problem_p02711_2():
    input_content = "123"
    expected_output = "No"
    run_pie_test_case("../p02711.py", input_content, expected_output)


def test_problem_p02711_3():
    input_content = "117"
    expected_output = "Yes"
    run_pie_test_case("../p02711.py", input_content, expected_output)
