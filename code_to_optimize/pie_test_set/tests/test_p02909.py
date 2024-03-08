from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02909_0():
    input_content = "Sunny"
    expected_output = "Cloudy"
    run_pie_test_case("../p02909.py", input_content, expected_output)


def test_problem_p02909_1():
    input_content = "Rainy"
    expected_output = "Sunny"
    run_pie_test_case("../p02909.py", input_content, expected_output)


def test_problem_p02909_2():
    input_content = "Sunny"
    expected_output = "Cloudy"
    run_pie_test_case("../p02909.py", input_content, expected_output)
