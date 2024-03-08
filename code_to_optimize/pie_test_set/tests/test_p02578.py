from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02578_0():
    input_content = "5\n2 1 5 4 3"
    expected_output = "4"
    run_pie_test_case("../p02578.py", input_content, expected_output)


def test_problem_p02578_1():
    input_content = "5\n3 3 3 3 3"
    expected_output = "0"
    run_pie_test_case("../p02578.py", input_content, expected_output)


def test_problem_p02578_2():
    input_content = "5\n2 1 5 4 3"
    expected_output = "4"
    run_pie_test_case("../p02578.py", input_content, expected_output)
