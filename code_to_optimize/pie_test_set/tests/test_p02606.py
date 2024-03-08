from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02606_0():
    input_content = "5 10 2"
    expected_output = "3"
    run_pie_test_case("../p02606.py", input_content, expected_output)


def test_problem_p02606_1():
    input_content = "1 100 1"
    expected_output = "100"
    run_pie_test_case("../p02606.py", input_content, expected_output)


def test_problem_p02606_2():
    input_content = "6 20 7"
    expected_output = "2"
    run_pie_test_case("../p02606.py", input_content, expected_output)


def test_problem_p02606_3():
    input_content = "5 10 2"
    expected_output = "3"
    run_pie_test_case("../p02606.py", input_content, expected_output)
