from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02720_0():
    input_content = "15"
    expected_output = "23"
    run_pie_test_case("../p02720.py", input_content, expected_output)


def test_problem_p02720_1():
    input_content = "13"
    expected_output = "21"
    run_pie_test_case("../p02720.py", input_content, expected_output)


def test_problem_p02720_2():
    input_content = "1"
    expected_output = "1"
    run_pie_test_case("../p02720.py", input_content, expected_output)


def test_problem_p02720_3():
    input_content = "15"
    expected_output = "23"
    run_pie_test_case("../p02720.py", input_content, expected_output)


def test_problem_p02720_4():
    input_content = "100000"
    expected_output = "3234566667"
    run_pie_test_case("../p02720.py", input_content, expected_output)
