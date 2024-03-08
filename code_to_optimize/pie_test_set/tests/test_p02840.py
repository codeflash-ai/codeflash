from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02840_0():
    input_content = "3 4 2"
    expected_output = "8"
    run_pie_test_case("../p02840.py", input_content, expected_output)


def test_problem_p02840_1():
    input_content = "2 3 -3"
    expected_output = "2"
    run_pie_test_case("../p02840.py", input_content, expected_output)


def test_problem_p02840_2():
    input_content = "3 4 2"
    expected_output = "8"
    run_pie_test_case("../p02840.py", input_content, expected_output)


def test_problem_p02840_3():
    input_content = "100 14 20"
    expected_output = "49805"
    run_pie_test_case("../p02840.py", input_content, expected_output)
