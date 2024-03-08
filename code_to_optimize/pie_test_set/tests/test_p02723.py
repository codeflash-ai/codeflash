from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02723_0():
    input_content = "sippuu"
    expected_output = "Yes"
    run_pie_test_case("../p02723.py", input_content, expected_output)


def test_problem_p02723_1():
    input_content = "coffee"
    expected_output = "Yes"
    run_pie_test_case("../p02723.py", input_content, expected_output)


def test_problem_p02723_2():
    input_content = "sippuu"
    expected_output = "Yes"
    run_pie_test_case("../p02723.py", input_content, expected_output)


def test_problem_p02723_3():
    input_content = "iphone"
    expected_output = "No"
    run_pie_test_case("../p02723.py", input_content, expected_output)
