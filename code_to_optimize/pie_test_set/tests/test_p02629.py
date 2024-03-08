from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02629_0():
    input_content = "2"
    expected_output = "b"
    run_pie_test_case("../p02629.py", input_content, expected_output)


def test_problem_p02629_1():
    input_content = "27"
    expected_output = "aa"
    run_pie_test_case("../p02629.py", input_content, expected_output)


def test_problem_p02629_2():
    input_content = "2"
    expected_output = "b"
    run_pie_test_case("../p02629.py", input_content, expected_output)


def test_problem_p02629_3():
    input_content = "123456789"
    expected_output = "jjddja"
    run_pie_test_case("../p02629.py", input_content, expected_output)
