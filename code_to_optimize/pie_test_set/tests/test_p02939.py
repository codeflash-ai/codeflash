from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02939_0():
    input_content = "aabbaa"
    expected_output = "4"
    run_pie_test_case("../p02939.py", input_content, expected_output)


def test_problem_p02939_1():
    input_content = "aaaccacabaababc"
    expected_output = "12"
    run_pie_test_case("../p02939.py", input_content, expected_output)


def test_problem_p02939_2():
    input_content = "aabbaa"
    expected_output = "4"
    run_pie_test_case("../p02939.py", input_content, expected_output)
