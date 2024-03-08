from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02766_0():
    input_content = "11 2"
    expected_output = "4"
    run_pie_test_case("../p02766.py", input_content, expected_output)


def test_problem_p02766_1():
    input_content = "11 2"
    expected_output = "4"
    run_pie_test_case("../p02766.py", input_content, expected_output)


def test_problem_p02766_2():
    input_content = "314159265 3"
    expected_output = "18"
    run_pie_test_case("../p02766.py", input_content, expected_output)


def test_problem_p02766_3():
    input_content = "1010101 10"
    expected_output = "7"
    run_pie_test_case("../p02766.py", input_content, expected_output)
