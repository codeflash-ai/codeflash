from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02973_0():
    input_content = "5\n2\n1\n4\n5\n3"
    expected_output = "2"
    run_pie_test_case("../p02973.py", input_content, expected_output)


def test_problem_p02973_1():
    input_content = "4\n0\n0\n0\n0"
    expected_output = "4"
    run_pie_test_case("../p02973.py", input_content, expected_output)


def test_problem_p02973_2():
    input_content = "5\n2\n1\n4\n5\n3"
    expected_output = "2"
    run_pie_test_case("../p02973.py", input_content, expected_output)
