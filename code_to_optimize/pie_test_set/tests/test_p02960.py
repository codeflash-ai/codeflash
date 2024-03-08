from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02960_0():
    input_content = "??2??5"
    expected_output = "768"
    run_pie_test_case("../p02960.py", input_content, expected_output)


def test_problem_p02960_1():
    input_content = "7?4"
    expected_output = "0"
    run_pie_test_case("../p02960.py", input_content, expected_output)


def test_problem_p02960_2():
    input_content = "??2??5"
    expected_output = "768"
    run_pie_test_case("../p02960.py", input_content, expected_output)


def test_problem_p02960_3():
    input_content = "?6?42???8??2??06243????9??3???7258??5??7???????774????4?1??17???9?5?70???76???"
    expected_output = "153716888"
    run_pie_test_case("../p02960.py", input_content, expected_output)


def test_problem_p02960_4():
    input_content = "?44"
    expected_output = "1"
    run_pie_test_case("../p02960.py", input_content, expected_output)
