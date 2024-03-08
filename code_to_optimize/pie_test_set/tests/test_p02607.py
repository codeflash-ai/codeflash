from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02607_0():
    input_content = "5\n1 3 4 5 7"
    expected_output = "2"
    run_pie_test_case("../p02607.py", input_content, expected_output)


def test_problem_p02607_1():
    input_content = "15\n13 76 46 15 50 98 93 77 31 43 84 90 6 24 14"
    expected_output = "3"
    run_pie_test_case("../p02607.py", input_content, expected_output)


def test_problem_p02607_2():
    input_content = "5\n1 3 4 5 7"
    expected_output = "2"
    run_pie_test_case("../p02607.py", input_content, expected_output)
