from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02755_0():
    input_content = "2 2"
    expected_output = "25"
    run_pie_test_case("../p02755.py", input_content, expected_output)


def test_problem_p02755_1():
    input_content = "19 99"
    expected_output = "-1"
    run_pie_test_case("../p02755.py", input_content, expected_output)


def test_problem_p02755_2():
    input_content = "2 2"
    expected_output = "25"
    run_pie_test_case("../p02755.py", input_content, expected_output)


def test_problem_p02755_3():
    input_content = "8 10"
    expected_output = "100"
    run_pie_test_case("../p02755.py", input_content, expected_output)
