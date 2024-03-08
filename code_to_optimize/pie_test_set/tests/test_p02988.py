from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02988_0():
    input_content = "5\n1 3 5 4 2"
    expected_output = "2"
    run_pie_test_case("../p02988.py", input_content, expected_output)


def test_problem_p02988_1():
    input_content = "9\n9 6 3 2 5 8 7 4 1"
    expected_output = "5"
    run_pie_test_case("../p02988.py", input_content, expected_output)


def test_problem_p02988_2():
    input_content = "5\n1 3 5 4 2"
    expected_output = "2"
    run_pie_test_case("../p02988.py", input_content, expected_output)
