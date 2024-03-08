from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02682_0():
    input_content = "2 1 1 3"
    expected_output = "2"
    run_pie_test_case("../p02682.py", input_content, expected_output)


def test_problem_p02682_1():
    input_content = "2000000000 0 0 2000000000"
    expected_output = "2000000000"
    run_pie_test_case("../p02682.py", input_content, expected_output)


def test_problem_p02682_2():
    input_content = "1 2 3 4"
    expected_output = "0"
    run_pie_test_case("../p02682.py", input_content, expected_output)


def test_problem_p02682_3():
    input_content = "2 1 1 3"
    expected_output = "2"
    run_pie_test_case("../p02682.py", input_content, expected_output)
