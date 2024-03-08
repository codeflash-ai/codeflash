from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02278_0():
    input_content = "5\n1 5 3 4 2"
    expected_output = "7"
    run_pie_test_case("../p02278.py", input_content, expected_output)


def test_problem_p02278_1():
    input_content = "4\n4 3 2 1"
    expected_output = "10"
    run_pie_test_case("../p02278.py", input_content, expected_output)


def test_problem_p02278_2():
    input_content = "5\n1 5 3 4 2"
    expected_output = "7"
    run_pie_test_case("../p02278.py", input_content, expected_output)
