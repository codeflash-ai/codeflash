from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02772_0():
    input_content = "5\n6 7 9 10 31"
    expected_output = "APPROVED"
    run_pie_test_case("../p02772.py", input_content, expected_output)


def test_problem_p02772_1():
    input_content = "5\n6 7 9 10 31"
    expected_output = "APPROVED"
    run_pie_test_case("../p02772.py", input_content, expected_output)


def test_problem_p02772_2():
    input_content = "3\n28 27 24"
    expected_output = "DENIED"
    run_pie_test_case("../p02772.py", input_content, expected_output)
