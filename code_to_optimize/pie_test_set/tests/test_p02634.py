from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02634_0():
    input_content = "1 1 2 2"
    expected_output = "3"
    run_pie_test_case("../p02634.py", input_content, expected_output)


def test_problem_p02634_1():
    input_content = "2 1 3 4"
    expected_output = "65"
    run_pie_test_case("../p02634.py", input_content, expected_output)


def test_problem_p02634_2():
    input_content = "31 41 59 265"
    expected_output = "387222020"
    run_pie_test_case("../p02634.py", input_content, expected_output)


def test_problem_p02634_3():
    input_content = "1 1 2 2"
    expected_output = "3"
    run_pie_test_case("../p02634.py", input_content, expected_output)
