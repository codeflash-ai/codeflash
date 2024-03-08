from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03139_0():
    input_content = "10 3 5"
    expected_output = "3 0"
    run_pie_test_case("../p03139.py", input_content, expected_output)


def test_problem_p03139_1():
    input_content = "10 7 5"
    expected_output = "5 2"
    run_pie_test_case("../p03139.py", input_content, expected_output)


def test_problem_p03139_2():
    input_content = "100 100 100"
    expected_output = "100 100"
    run_pie_test_case("../p03139.py", input_content, expected_output)


def test_problem_p03139_3():
    input_content = "10 3 5"
    expected_output = "3 0"
    run_pie_test_case("../p03139.py", input_content, expected_output)
