from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03419_0():
    input_content = "2 2"
    expected_output = "0"
    run_pie_test_case("../p03419.py", input_content, expected_output)


def test_problem_p03419_1():
    input_content = "1 7"
    expected_output = "5"
    run_pie_test_case("../p03419.py", input_content, expected_output)


def test_problem_p03419_2():
    input_content = "2 2"
    expected_output = "0"
    run_pie_test_case("../p03419.py", input_content, expected_output)


def test_problem_p03419_3():
    input_content = "314 1592"
    expected_output = "496080"
    run_pie_test_case("../p03419.py", input_content, expected_output)
