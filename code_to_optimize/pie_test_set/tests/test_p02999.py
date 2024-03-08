from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02999_0():
    input_content = "3 5"
    expected_output = "0"
    run_pie_test_case("../p02999.py", input_content, expected_output)


def test_problem_p02999_1():
    input_content = "6 6"
    expected_output = "10"
    run_pie_test_case("../p02999.py", input_content, expected_output)


def test_problem_p02999_2():
    input_content = "7 5"
    expected_output = "10"
    run_pie_test_case("../p02999.py", input_content, expected_output)


def test_problem_p02999_3():
    input_content = "3 5"
    expected_output = "0"
    run_pie_test_case("../p02999.py", input_content, expected_output)
