from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02981_0():
    input_content = "4 2 9"
    expected_output = "8"
    run_pie_test_case("../p02981.py", input_content, expected_output)


def test_problem_p02981_1():
    input_content = "4 2 7"
    expected_output = "7"
    run_pie_test_case("../p02981.py", input_content, expected_output)


def test_problem_p02981_2():
    input_content = "4 2 9"
    expected_output = "8"
    run_pie_test_case("../p02981.py", input_content, expected_output)


def test_problem_p02981_3():
    input_content = "4 2 8"
    expected_output = "8"
    run_pie_test_case("../p02981.py", input_content, expected_output)
