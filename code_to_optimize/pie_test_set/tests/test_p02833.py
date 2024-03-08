from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02833_0():
    input_content = "12"
    expected_output = "1"
    run_pie_test_case("../p02833.py", input_content, expected_output)


def test_problem_p02833_1():
    input_content = "1000000000000000000"
    expected_output = "124999999999999995"
    run_pie_test_case("../p02833.py", input_content, expected_output)


def test_problem_p02833_2():
    input_content = "5"
    expected_output = "0"
    run_pie_test_case("../p02833.py", input_content, expected_output)


def test_problem_p02833_3():
    input_content = "12"
    expected_output = "1"
    run_pie_test_case("../p02833.py", input_content, expected_output)
