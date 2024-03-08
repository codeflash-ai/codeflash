from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02697_0():
    input_content = "4 1"
    expected_output = "2 3"
    run_pie_test_case("../p02697.py", input_content, expected_output)


def test_problem_p02697_1():
    input_content = "7 3"
    expected_output = "1 6\n2 5\n3 4"
    run_pie_test_case("../p02697.py", input_content, expected_output)


def test_problem_p02697_2():
    input_content = "4 1"
    expected_output = "2 3"
    run_pie_test_case("../p02697.py", input_content, expected_output)
