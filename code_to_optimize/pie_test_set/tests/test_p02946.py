from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02946_0():
    input_content = "3 7"
    expected_output = "5 6 7 8 9"
    run_pie_test_case("../p02946.py", input_content, expected_output)


def test_problem_p02946_1():
    input_content = "4 0"
    expected_output = "-3 -2 -1 0 1 2 3"
    run_pie_test_case("../p02946.py", input_content, expected_output)


def test_problem_p02946_2():
    input_content = "1 100"
    expected_output = "100"
    run_pie_test_case("../p02946.py", input_content, expected_output)


def test_problem_p02946_3():
    input_content = "3 7"
    expected_output = "5 6 7 8 9"
    run_pie_test_case("../p02946.py", input_content, expected_output)
