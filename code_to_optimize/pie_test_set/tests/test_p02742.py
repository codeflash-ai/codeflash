from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02742_0():
    input_content = "4 5"
    expected_output = "10"
    run_pie_test_case("../p02742.py", input_content, expected_output)


def test_problem_p02742_1():
    input_content = "7 3"
    expected_output = "11"
    run_pie_test_case("../p02742.py", input_content, expected_output)


def test_problem_p02742_2():
    input_content = "1000000000 1000000000"
    expected_output = "500000000000000000"
    run_pie_test_case("../p02742.py", input_content, expected_output)


def test_problem_p02742_3():
    input_content = "4 5"
    expected_output = "10"
    run_pie_test_case("../p02742.py", input_content, expected_output)
