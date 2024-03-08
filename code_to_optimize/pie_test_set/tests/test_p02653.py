from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02653_0():
    input_content = "4 2 3"
    expected_output = "11"
    run_pie_test_case("../p02653.py", input_content, expected_output)


def test_problem_p02653_1():
    input_content = "4 2 3"
    expected_output = "11"
    run_pie_test_case("../p02653.py", input_content, expected_output)


def test_problem_p02653_2():
    input_content = "10 7 2"
    expected_output = "533"
    run_pie_test_case("../p02653.py", input_content, expected_output)


def test_problem_p02653_3():
    input_content = "1000 100 10"
    expected_output = "828178524"
    run_pie_test_case("../p02653.py", input_content, expected_output)
