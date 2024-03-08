from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02927_0():
    input_content = "15 40"
    expected_output = "10"
    run_pie_test_case("../p02927.py", input_content, expected_output)


def test_problem_p02927_1():
    input_content = "15 40"
    expected_output = "10"
    run_pie_test_case("../p02927.py", input_content, expected_output)


def test_problem_p02927_2():
    input_content = "1 1"
    expected_output = "0"
    run_pie_test_case("../p02927.py", input_content, expected_output)


def test_problem_p02927_3():
    input_content = "12 31"
    expected_output = "5"
    run_pie_test_case("../p02927.py", input_content, expected_output)
