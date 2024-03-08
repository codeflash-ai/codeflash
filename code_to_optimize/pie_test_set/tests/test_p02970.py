from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02970_0():
    input_content = "6 2"
    expected_output = "2"
    run_pie_test_case("../p02970.py", input_content, expected_output)


def test_problem_p02970_1():
    input_content = "20 4"
    expected_output = "3"
    run_pie_test_case("../p02970.py", input_content, expected_output)


def test_problem_p02970_2():
    input_content = "14 3"
    expected_output = "2"
    run_pie_test_case("../p02970.py", input_content, expected_output)


def test_problem_p02970_3():
    input_content = "6 2"
    expected_output = "2"
    run_pie_test_case("../p02970.py", input_content, expected_output)
