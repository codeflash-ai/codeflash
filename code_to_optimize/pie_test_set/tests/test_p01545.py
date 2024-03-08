from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01545_0():
    input_content = "4\n1 4 2 3"
    expected_output = "4"
    run_pie_test_case("../p01545.py", input_content, expected_output)


def test_problem_p01545_1():
    input_content = "8\n6 2 1 3 8 5 4 7"
    expected_output = "19"
    run_pie_test_case("../p01545.py", input_content, expected_output)


def test_problem_p01545_2():
    input_content = "5\n1 5 3 2 4"
    expected_output = "7"
    run_pie_test_case("../p01545.py", input_content, expected_output)


def test_problem_p01545_3():
    input_content = "7\n1 2 3 4 5 6 7"
    expected_output = "0"
    run_pie_test_case("../p01545.py", input_content, expected_output)


def test_problem_p01545_4():
    input_content = "4\n1 4 2 3"
    expected_output = "4"
    run_pie_test_case("../p01545.py", input_content, expected_output)
