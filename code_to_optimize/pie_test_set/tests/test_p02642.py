from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02642_0():
    input_content = "5\n24 11 8 3 16"
    expected_output = "3"
    run_pie_test_case("../p02642.py", input_content, expected_output)


def test_problem_p02642_1():
    input_content = "5\n24 11 8 3 16"
    expected_output = "3"
    run_pie_test_case("../p02642.py", input_content, expected_output)


def test_problem_p02642_2():
    input_content = "10\n33 18 45 28 8 19 89 86 2 4"
    expected_output = "5"
    run_pie_test_case("../p02642.py", input_content, expected_output)


def test_problem_p02642_3():
    input_content = "4\n5 5 5 5"
    expected_output = "0"
    run_pie_test_case("../p02642.py", input_content, expected_output)
