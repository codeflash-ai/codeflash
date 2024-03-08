from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02574_0():
    input_content = "3\n3 4 5"
    expected_output = "pairwise coprime"
    run_pie_test_case("../p02574.py", input_content, expected_output)


def test_problem_p02574_1():
    input_content = "3\n6 10 16"
    expected_output = "not coprime"
    run_pie_test_case("../p02574.py", input_content, expected_output)


def test_problem_p02574_2():
    input_content = "3\n6 10 15"
    expected_output = "setwise coprime"
    run_pie_test_case("../p02574.py", input_content, expected_output)


def test_problem_p02574_3():
    input_content = "3\n3 4 5"
    expected_output = "pairwise coprime"
    run_pie_test_case("../p02574.py", input_content, expected_output)
