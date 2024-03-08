from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01751_0():
    input_content = "10 10 5"
    expected_output = "5"
    run_pie_test_case("../p01751.py", input_content, expected_output)


def test_problem_p01751_1():
    input_content = "20 20 20"
    expected_output = "20"
    run_pie_test_case("../p01751.py", input_content, expected_output)


def test_problem_p01751_2():
    input_content = "50 40 51"
    expected_output = "111"
    run_pie_test_case("../p01751.py", input_content, expected_output)


def test_problem_p01751_3():
    input_content = "10 10 5"
    expected_output = "5"
    run_pie_test_case("../p01751.py", input_content, expected_output)


def test_problem_p01751_4():
    input_content = "30 30 40"
    expected_output = "-1"
    run_pie_test_case("../p01751.py", input_content, expected_output)
