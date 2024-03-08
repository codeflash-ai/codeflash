from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02719_0():
    input_content = "7 4"
    expected_output = "1"
    run_pie_test_case("../p02719.py", input_content, expected_output)


def test_problem_p02719_1():
    input_content = "1000000000000000000 1"
    expected_output = "0"
    run_pie_test_case("../p02719.py", input_content, expected_output)


def test_problem_p02719_2():
    input_content = "7 4"
    expected_output = "1"
    run_pie_test_case("../p02719.py", input_content, expected_output)


def test_problem_p02719_3():
    input_content = "2 6"
    expected_output = "2"
    run_pie_test_case("../p02719.py", input_content, expected_output)
