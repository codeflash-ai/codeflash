from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02659_0():
    input_content = "198 1.10"
    expected_output = "217"
    run_pie_test_case("../p02659.py", input_content, expected_output)


def test_problem_p02659_1():
    input_content = "1000000000000000 9.99"
    expected_output = "9990000000000000"
    run_pie_test_case("../p02659.py", input_content, expected_output)


def test_problem_p02659_2():
    input_content = "1 0.01"
    expected_output = "0"
    run_pie_test_case("../p02659.py", input_content, expected_output)


def test_problem_p02659_3():
    input_content = "198 1.10"
    expected_output = "217"
    run_pie_test_case("../p02659.py", input_content, expected_output)
