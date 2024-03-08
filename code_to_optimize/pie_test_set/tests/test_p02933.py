from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02933_0():
    input_content = "3200\npink"
    expected_output = "pink"
    run_pie_test_case("../p02933.py", input_content, expected_output)


def test_problem_p02933_1():
    input_content = "4049\nred"
    expected_output = "red"
    run_pie_test_case("../p02933.py", input_content, expected_output)


def test_problem_p02933_2():
    input_content = "3199\npink"
    expected_output = "red"
    run_pie_test_case("../p02933.py", input_content, expected_output)


def test_problem_p02933_3():
    input_content = "3200\npink"
    expected_output = "pink"
    run_pie_test_case("../p02933.py", input_content, expected_output)
